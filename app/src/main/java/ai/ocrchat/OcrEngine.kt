package ai.ocrchat

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Строка, найденная на странице. */
data class OcrLine(val x: Int, val y: Int, val w: Int, val h: Int, val text: String, val conf: Float)
data class OcrResult(val lines: List<OcrLine>, val annotated: Bitmap, val ms: Long)

/**
 * Пайплайн = порт OnnxRu из приложения Yomikai:
 * luma<170 → морфозамыкание 25×3 ×2 → компоненты (≥10В/12Ш) → группировка в строки.
 * Тёмные страницы (luma<100): инверсный порог + комикс-баллоны (сырые блобы l<45/l>210,
 * fill>=45%, не у края -> строки-полосы rowsum; кольца контуров и шум режутся BFS);
 * яркость в readLine инвертируется только для полярности своей строки.
 * readLine: h=32, белый правый пад, luma/255, CTC blank=109/unk=110, softmax-медиана.
 * Все кропы жёстко клампятся в рамки битмапа (защита от "empty range / max<min").
 */
class OcrEngine(modelBytes: ByteArray) {

    companion object {
        const val CHARSET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" +
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" +
            "ЇїІіЄєҐґ" +
            " .,!?;:\"'«»„“”()[]—–-…№*/0123456789"
        const val BLANK = 109
        const val UNK = 110
        const val LINE_H = 32
        const val MAX_DIM = 1400
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
        inputName = session.inputNames.first()
    }

    val description: String
        get() = "in=$inputName ${session.inputInfo[inputName]?.info?.let { (it as ai.onnxruntime.TensorInfo).shape.contentToString() }}, out=${session.outputNames.first()}"

    fun close() = session.close()

    // ---------- публичный вход ----------
    fun run(src: Bitmap): OcrResult {
        val t0 = System.currentTimeMillis()
        val sc = min(1f, MAX_DIM.toFloat() / max(src.width, src.height))
        val page = if (sc < 1f) Bitmap.createScaledBitmap(src, max(1, (src.width * sc).roundToInt()), max(1, (src.height * sc).roundToInt()), true) else src
        val w = page.width
        val h = page.height
        val px = IntArray(w * h)
        page.getPixels(px, 0, w, 0, 0, w, h)

        // авто-детект тёмной страницы: сэмплим каждый 7-й пиксель
        var sum = 0L; var n = 0
        var i = 0
        while (i < px.size) {
            val c = px[i]
            sum += (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000
            n++
            i += 7
        }
        val dark = n > 0 && (sum / n) < 100

        val boxes = segmentAny(px, w, h, dark)
        val lines = ArrayList<OcrLine>(boxes.size)
        for (b in boxes) {
            val line = readLine(page, b[0], b[1], b[2], b[3], b[4] == 1) ?: continue
            if (line.text.isNotBlank()) lines.add(line)
        }
        return OcrResult(lines, annotate(page, lines), System.currentTimeMillis() - t0)
    }

    // ---------- сегментация ----------
    private fun segment(px: IntArray, w: Int, h: Int, dark: Boolean): List<IntArray> {
        var mask = ByteArray(w * h)
        for (i in px.indices) {
            val c = px[i]
            val l = (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000
            // тёмная страница: текст светлее фона → инвертированный порог
            mask[i] = if (dark) (if (l > 127) 1 else 0) else (if (l < 170) 1 else 0)
        }
        repeat(2) {
            mask = erode(dilate(mask, w, h, 12, 1), w, h, 12, 1) // close 25x3
        }
        val comps = components(mask, w, h)
        // группировка в строки по позиции центра
        val rows = ArrayList<ArrayList<IntArray>>()
        for (b in comps.sortedWith(compareBy({ it[1] }, { it[0] }))) {
            val bh = b[3]
            val cy = b[1] + bh / 2f
            var placed = false
            for (row in rows) {
                val ry0 = row.minOf { it[1] }.toFloat()
                val ry1 = row.maxOf { it[1] + it[3] }.toFloat()
                if (ry0 - 0.8f * bh <= cy && cy <= ry1 + 0.8f * bh) { row.add(b); placed = true; break }
            }
            if (!placed) rows.add(arrayListOf(b))
        }
        rows.sortBy { row -> row.minOf { it[1] } }
        val out = ArrayList<IntArray>(rows.size)
        for (row in rows) {
            row.sortBy { it[0] }
            val x0 = row.minOf { it[0] }
            val y0 = row.minOf { it[1] }
            val x1 = row.maxOf { it[0] + it[2] }
            val y1 = row.maxOf { it[1] + it[3] }
            // КЛАМП в рамки (важно): иначе «Cannot coerce value to an empty range»
            val cx0 = x0.coerceIn(0, w - 1)
            val cy0 = y0.coerceIn(0, h - 1)
            val cx1 = x1.coerceIn(1, w)
            val cy1 = y1.coerceIn(1, h)
            if (cx1 > cx0 && cy1 > cy0) out.add(intArrayOf(cx0, cy0, cx1 - cx0, cy1 - cy0))
        }
        return out
    }

    private fun dilate(m: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        val tmp = ByteArray(w * h)
        for (y in 0 until h) {
            val row = y * w
            var run = 0
            for (x in 0 until w + rx) {
                if (x < w) run += mask(m, row + x)
                if (x > rx * 2) run -= mask(m, row + x - rx * 2 - 1)
                if (x >= rx && run > 0) tmp[row + x - rx] = 1
            }
        }
        val out = ByteArray(w * h)
        for (x in 0 until w) {
            var run = 0
            for (y in 0 until h + ry) {
                if (y < h) run += tmp[y * w + x].toInt()
                if (y > ry * 2) run -= tmp[(y - ry * 2 - 1) * w + x].toInt()
                if (y >= ry && run > 0) out[(y - ry) * w + x] = 1
            }
        }
        return out
    }

    private fun erode(m: ByteArray, w: Int, h: Int, rx: Int, ry: Int): ByteArray {
        val inv = ByteArray(w * h) { i -> (1 - m[i]).toByte() }
        val d = dilate(inv, w, h, rx, ry)
        return ByteArray(w * h) { i -> (1 - d[i]).toByte() }
    }

    private fun mask(m: ByteArray, i: Int) = m[i].toInt()

    private fun lumaOf(c: Int) = (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000

    // ---------- v1.4: комикс-баллоны на тёмных страницах ----------
    /** Обёртка: светлая страница = v1.1-сегментация; тёмная = баллоны + остаточный проход.
     *  Каждый бокс имеет 5-й элемент: полярность (1 = светлый текст, инвертировать яркость). */
    private fun segmentAny(px: IntArray, w: Int, h: Int, dark: Boolean): List<IntArray> {
        if (!dark) return segment(px, w, h, false).map { intArrayOf(it[0], it[1], it[2], it[3], 0) }
        val out = ArrayList<IntArray>()
        val bubbles = findBubbles(px, w, h, true) + findBubbles(px, w, h, false)
        for (b in bubbles) {
            // внутренняя рамка баллона (отступ от контура), все клампы!
            val ix = min(b[0] + 8, w - 1); val iy = min(b[1] + 8, h - 1)
            val iw = min(b[2] - 16, w - ix); val ih = min(b[3] - 16, h - iy)
            if (iw < 30 || ih < 16) continue
            val sub = IntArray(iw * ih)
            for (yy in 0 until ih) System.arraycopy(px, (iy + yy) * w + ix, sub, yy * iw, iw)
            val pol = b[4] == 1
            for (r in bubbleLines(sub, iw, ih, pol)) {
                // лёгкий паддинг, чтобы не резать первую/последнюю букву
                val x0 = max(ix, ix + r[0] - 2); val y0 = max(iy, iy + r[1] - 1)
                val x1 = min(ix + iw, ix + r[0] + r[2] + 2); val y1 = min(iy + ih, iy + r[1] + r[3] + 1)
                out.add(intArrayOf(x0, y0, x1 - x0, y1 - y0, if (pol) 1 else 0))
            }
        }
        // остаточный полностраничный проход (статус-бары и пр.), минус гигантские склейки арта
        for (b in segment(px, w, h, true)) {
            if (b[3] > h * 6 / 100 || b[2] * b[3] * 12 > w * h) continue
            val cx = b[0] + b[2] / 2; val cy = b[1] + b[3] / 2
            var inside = false
            for (bb in bubbles) {
                if (cx >= bb[0] && cx < bb[0] + bb[2] && cy >= bb[1] && cy < bb[1] + bb[3]) { inside = true; break }
            }
            if (!inside) out.add(intArrayOf(b[0], b[1], b[2], b[3], 1))
        }
        out.sortWith(compareBy({ it[1] }, { it[0] }))
        return out
    }

    /** Однотонные баллоны: СЫРЫЕ компоненты (замыкание тут вредно — склеивает с бордюрами
     *  скриншота), fill≥45%, размеры 2–50% страницы, не касаются края страницы. */
    private fun findBubbles(px: IntArray, w: Int, h: Int, darkBlob: Boolean): List<IntArray> {
        val raw = ByteArray(w * h)
        for (i in px.indices) {
            val l = lumaOf(px[i])
            raw[i] = if (darkBlob) (if (l < 45) 1 else 0) else (if (l > 210) 1 else 0)
        }
        val out = ArrayList<IntArray>()
        for (b in componentsCnt(raw, w, h)) {
            val bw = b[2]; val bh = b[3]; val cnt = b[4]
            val area = bw * bh
            if (bw < 60 || bh < 40 || area < w * h / 50 || area * 2 > w * h) continue
            if (cnt < area * 45 / 100) continue
            if (b[0] == 0 || b[1] == 0 || b[0] + bw == w || b[1] + bh == h) continue
            out.add(intArrayOf(b[0], b[1], bw, bh, if (darkBlob) 1 else 0))
        }
        return out
    }

    /** components() с количеством пикселей в [4]-м элементе. */
    private fun componentsCnt(maskArr: ByteArray, w: Int, h: Int): List<IntArray> {
        val lab = IntArray(w * h) { -1 }
        val q = IntArray(w * h)
        val out = ArrayList<IntArray>()
        var id = 0
        for (s in 0 until w * h) {
            if (maskArr[s].toInt() == 0 || lab[s] >= 0) continue
            var head = 0; var tail = 0
            q[tail++] = s; lab[s] = id
            var x0 = w; var x1 = 0; var y0 = h; var y1 = 0
            while (head < tail) {
                val idx = q[head++]
                val x = idx % w; val y = idx / w
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
                    val ni = ny * w + nx
                    if (maskArr[ni].toInt() == 1 && lab[ni] < 0) { lab[ni] = id; q[tail++] = ni }
                }
            }
            val bw = x1 - x0 + 1; val bh = y1 - y0 + 1
            out.add(intArrayOf(x0, y0, bw, bh, tail))
            if (out.size >= 400) return out
            id++
        }
        return out
    }

    /** Строки внутри баллона: BFS зануляет компоненты, касающиеся края (светлое кольцо
     *  контура) и мусор <3px (JPEG-шум), затем строки = горизонтальные полосы по rowsum. */
    private fun bubbleLines(sub: IntArray, sw: Int, sh: Int, polDark: Boolean): List<IntArray> {
        val m = ByteArray(sw * sh)
        for (i in sub.indices) {
            val l = lumaOf(sub[i])
            m[i] = if (polDark) (if (l > 127) 1 else 0) else (if (l < 127) 1 else 0)
        }
        val lab = ByteArray(sw * sh)
        val q = IntArray(sw * sh)
        for (s0 in 0 until sw * sh) {
            if (m[s0].toInt() == 0 || lab[s0].toInt() != 0) continue
            var head = 0; var tail = 0
            q[tail++] = s0; lab[s0] = 1
            var touch = false
            var cnt = 0
            val pts = ArrayList<Int>()
            while (head < tail) {
                val idx = q[head++]; pts.add(idx); cnt++
                val cx = idx % sw; val cy = idx / sw
                if (cy == 0 || cx == 0 || cy == sh - 1 || cx == sw - 1) touch = true
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = cx + dx; val ny = cy + dy
                    if (nx < 0 || ny < 0 || nx >= sw || ny >= sh) continue
                    val ni = ny * sw + nx
                    if (m[ni].toInt() == 1 && lab[ni].toInt() == 0) { lab[ni] = 1; q[tail++] = ni }
                }
            }
            if (touch || cnt < 3) for (p in pts) m[p] = 0
        }
        val rowsum = IntArray(sh)
        for (y in 0 until sh) { var s2 = 0; val off = y * sw; for (x in 0 until sw) s2 += m[off + x].toInt(); rowsum[y] = s2 }
        val rows = ArrayList<IntArray>()
        var s = -1
        for (i in 0..sh) {
            if (i < sh && rowsum[i] >= 2) { if (s < 0) s = i; continue }
            if (s >= 0) {
                val y0 = s; val y1 = i - 1; s = -1
                var x0 = sw; var x1 = -1
                for (y in y0..y1) {
                    val off = y * sw
                    for (x in 0 until sw) if (m[off + x].toInt() == 1) { if (x < x0) x0 = x; if (x > x1) x1 = x }
                }
                if (x1 < 0) continue
                val bh2 = y1 - y0 + 1; val bw2 = x1 - x0 + 1
                if (bh2 >= 6 && bh2 <= sh * 35 / 100 && bw2 >= 10) rows.add(intArrayOf(x0, y0, bw2, bh2))
            }
        }
        return rows
    }

    private fun components(mask: ByteArray, w: Int, h: Int): List<IntArray> {
        val lab = IntArray(w * h) { -1 }
        val q = IntArray(w * h)
        val out = ArrayList<IntArray>()
        var id = 0
        for (s in 0 until w * h) {
            if (mask[s].toInt() == 0 || lab[s] >= 0) continue
            var head = 0
            var tail = 0
            q[tail++] = s
            lab[s] = id
            var x0 = w; var x1 = 0; var y0 = h; var y1 = 0
            while (head < tail) {
                val idx = q[head++]
                val x = idx % w
                val y = idx / w
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
                for (dy in -1..1) for (dx in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
                    val ni = ny * w + nx
                    if (mask[ni].toInt() == 1 && lab[ni] < 0) { lab[ni] = id; q[tail++] = ni }
                }
            }
            val bw = x1 - x0 + 1
            val bh = y1 - y0 + 1
            if (bh >= 10 && bw >= 12) {
                out.add(intArrayOf(x0, y0, bw, bh))
                if (out.size >= 300) return out
            }
            id++
        }
        return out
    }

    // ---------- распознавание строки ----------
    private fun readLine(page: Bitmap, x: Int, y: Int, w: Int, h: Int, dark: Boolean): OcrLine? {
        // страховка от пустых/перевёрнутых диапазонов
        val sx = x.coerceIn(0, page.width - 1)
        val sy = y.coerceIn(0, page.height - 1)
        val sw = min(w, page.width - sx)
        val sh = min(h, page.height - sy)
        if (sw <= 0 || sh <= 0) return null

        var newW = max(8, (sw * LINE_H.toFloat() / sh).roundToInt())
        newW = min(448, newW)
        val paddedW = ceil(newW / 8f).toInt() * 8

        val line = Bitmap.createBitmap(paddedW, LINE_H, Bitmap.Config.ARGB_8888)
        val c = Canvas(line)
        c.drawColor(Color.WHITE)
        val dst = android.graphics.RectF(0f, 0f, newW.toFloat(), LINE_H.toFloat())
        c.drawBitmap(page, android.graphics.Rect(sx, sy, sx + sw, sy + sh), dst, null)

        val px = IntArray(LINE_H * paddedW)
        line.getPixels(px, 0, paddedW, 0, 0, paddedW, LINE_H)
        val inp = FloatArray(LINE_H * paddedW)
        for (i in px.indices) {
            val col = px[i]
            val lum = (Color.red(col) * 299 + Color.green(col) * 587 + Color.blue(col) * 114) / 255000f
            // тёмная страница: инвертируем яркость — сеть ждёт тёмный текст на светлом фоне;
            // белый правый пад (1.0) после инверсии остаётся корректным фоном
            inp[i] = if (dark) 1f - lum else lum
        }

        val shape = longArrayOf(1, 1, LINE_H.toLong(), paddedW.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(inp), shape).use { tensor ->
            val res = session.run(mapOf(inputName to tensor))
            res.use {
                val t = it[0] as OnnxTensor
                val raw = (t.value as Array<Array<FloatArray>>)[0] // [T][C]
                val classes = raw[0].size
                return decode(raw, classes)?.let { (text, conf) -> OcrLine(sx, sy, sw, sh, text, conf) }
            }
        }
    }

    private fun decode(raw: Array<FloatArray>, classes: Int): Pair<String, Float>? {
        val sb = StringBuilder()
        var prev = -2
        val probs = ArrayList<Float>(raw.size)
        for (step in raw) {
            var mx = Float.NEGATIVE_INFINITY
            for (k in 0 until classes) if (step[k] > mx) mx = step[k]
            var sum = 0f
            for (k in 0 until classes) sum += exp((step[k] - mx).toDouble()).toFloat()
            var bi = 0
            var bs = Float.NEGATIVE_INFINITY
            for (k in 0 until classes) if (step[k] > bs) { bs = step[k]; bi = k }
            if (bi == BLANK || bi == UNK) { prev = bi; continue }
            if (bi != prev && bi >= 0) {
                if (bi < CHARSET.length) sb.append(CHARSET[bi]) else sb.append('�')
                probs.add(exp((bs - mx).toDouble()).toFloat() / sum)
            }
            prev = bi
        }
        val conf = if (probs.isEmpty()) 0f else { probs.sort(); probs[probs.size / 2] }
        var t = sb.toString().trim()
        // лёгкая нормализация артефактов
        while (t.contains("……")) t = t.replace("……", "…")
        t = t.replace(Regex("\\s+"), " ")
        if (t.isBlank()) return null
        return t to conf
    }

    // ---------- аннотированная копия ----------
    private fun annotate(page: Bitmap, lines: List<OcrLine>): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            color = 0xFF39D353.toInt()
            strokeWidth = max(2f, page.width / 400f)
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            color = 0xFFFFD479.toInt()
            isAntiAlias = true
            textSize = 28f
            typeface = android.graphics.Typeface.MONOSPACE
        }
        val bgPaint = Paint().apply { color = 0xB3101827.toInt() }
        for (l in lines) {
            c.drawRect(l.x.toFloat(), l.y.toFloat(), (l.x + l.w).toFloat(), (l.y + l.h).toFloat(), boxPaint)
            textPaint.textSize = max(14f, l.h * 0.55f)
            val tw = textPaint.measureText(l.text)
            val ty = max(textPaint.textSize, l.y - 6f)
            c.drawRect(l.x.toFloat(), ty - textPaint.textSize, l.x + tw + 12f, ty + 6f, bgPaint)
            c.drawText(l.text, l.x + 6f, ty, textPaint)
        }
        return out
    }
}
