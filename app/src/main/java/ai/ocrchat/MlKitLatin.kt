package ai.ocrchat

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

/**
 * Латинский контур: ML Kit text-recognition (script=latin, bundled — работает офлайн).
 * Используем только для строк, где >=60% букв латинские; кириллица остаётся за нашим CRNN.
 */
object MlKitLatin {
    data class LatLine(val x: Int, val y: Int, val w: Int, val h: Int, val text: String)

    private val recognizer by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    private fun latinCount(s: String): Pair<Int, Int> {
        var lat = 0; var letters = 0
        for (c in s) if (c.isLetter()) {
            letters++
            if (c in 'A'..'Z' || c in 'a'..'z') lat++
        }
        return lat to letters
    }

    /** Строки с латиницей в координатах входного битмапа. */
    fun detect(page: Bitmap): List<LatLine> = try {
        val result = Tasks.await(recognizer.process(InputImage.fromBitmap(page, 0)))
        val out = ArrayList<LatLine>()
        for (block in result.textBlocks) for (line in block.lines) {
            val bb = line.boundingBox ?: continue
            val t = line.text.replace(Regex("\\s+"), " ").trim()
            if (t.isEmpty()) continue
            val (lat, letters) = latinCount(t)
            if (lat >= 2 && lat * 5 >= letters * 3) {
                out.add(LatLine(bb.left.coerceAtLeast(0), bb.top.coerceAtLeast(0), bb.width(), bb.height(), t))
            }
        }
        out
    } catch (e: Exception) {
        emptyList()
    }
}
