package ai.ocrchat

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var chat: RecyclerView
    private lateinit var status: TextView
    private lateinit var modelName: TextView
    private val adapter = ChatAdapter()
    private var engine: OcrEngine? = null

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { loadModelFromUri(it) }
    }

    private val pickImages = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        for (u in uris) runOcrOn(u)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        chat = findViewById(R.id.chat)
        status = findViewById(R.id.status)
        modelName = findViewById(R.id.modelName)

        chat.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        chat.adapter = adapter

        findViewById<Button>(R.id.btnModel).setOnClickListener {
            pickModel.launch(arrayOf("*/*"))
        }
        findViewById<Button>(R.id.btnAttach).setOnClickListener {
            pickImages.launch("image/*")
        }

        adapter.add(Msg.BotText("Привет! Это офлайн-OCR чат.\n📦 встроена дообученная модель r4 (кириллица), можешь сменить на любую .onnx.\n📎 прикрепи картинку — отвечу, что где написано."))

        // встроенная модель r4 — лениво при первом старте
        lifecycleScope.launch {
            val st = withContext(Dispatchers.Default) {
                try {
                    val bytes = assets.open("best.onnx").readBytes()
                    engine = OcrEngine(bytes)
                    null
                } catch (t: Throwable) { t.stackTraceToString() }
            }
            if (st == null) {
                modelName.text = "модель: r4 (встроенная)"
                adapter.add(Msg.BotText("✅ модель r4 поднята"))
            } else {
                modelName.text = "модель: ОШИБКА"
                adapter.add(Msg.BotText("Не поднялась встроенная модель:\n$st", isErr = true))
            }
        }
    }

    private fun fileName(uri: Uri): String = contentResolver.query(uri, null, null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) c.getString(i) else null
    } ?: uri.lastPathSegment ?: "file"

    private fun loadModelFromUri(uri: Uri) {
        val name = fileName(uri)
        adapter.add(Msg.BotText("📦 $name — гружу…"))
        lifecycleScope.launch {
            val st = withContext(Dispatchers.Default) {
                try {
                    val bytes = contentResolver.openInputStream(uri)?.readBytes()
                        ?: return@withContext "нет доступа к файлу"
                    val e = OcrEngine(bytes)
                    engine?.close()
                    engine = e
                    null
                } catch (t: Throwable) { t.stackTraceToString() }
            }
            if (st == null) {
                modelName.text = "модель: $name"
                adapter.add(Msg.BotText("✅ модель $name поднята: ${engine?.description}"))
            } else {
                adapter.add(Msg.BotText("❌ модель $name не поднялась:\n$st", isErr = true))
            }
        }
    }

    private fun runOcrOn(uri: Uri) {
        val eng = engine
        if (eng == null) { adapter.add(Msg.BotText("модель ещё не поднята — подожди или выбери другую 📦", isErr = true)); return }
        lifecycleScope.launch {
            val bmp: Any? = withContext(Dispatchers.Default) {
                try {
                    contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                } catch (t: Throwable) { t.stackTraceToString() }
            }
            when (bmp) {
                is Bitmap -> {
                    adapter.add(Msg.UserImage(bmp, fileName(uri)))
                    adapter.add(Msg.BotTyping)
                    chat.scrollToPosition(adapter.itemCount - 1)
                    val out: Pair<String, Bitmap?> = withContext(Dispatchers.Default) {
                        try {
                            val r = eng.run(bmp)
                            val sb = StringBuilder()
                            if (r.lines.isEmpty()) {
                                sb.append("(текст не найден)")
                            } else {
                                sb.append("Найдено за ${"%.1f".format(r.ms / 1000f)}с, строк: ${r.lines.size}\n")
                                r.lines.forEachIndexed { i, l ->
                                    sb.append("${i + 1}. (${l.x},${l.y}) ${l.text}  · ${"%.2f".format(l.conf)}\n")
                                }
                            }
                            Pair(sb.toString(), r.annotated)
                        } catch (t: Throwable) {
                            Pair("ОШИБКА:\n" + t.stackTraceToString(), null)
                        }
                    }
                    adapter.replaceTyping(Msg.BotText(out.first, isErr = out.first.startsWith("ОШИБКА")))
                    if (out.second != null) adapter.add(Msg.UserImage(out.second!!, "annotated"))
                    chat.scrollToPosition(adapter.itemCount - 1)
                }
                else -> adapter.add(Msg.BotText("❌ не смог прочитать файл как картинку:\n$bmp", isErr = true))
            }
        }
    }
}
