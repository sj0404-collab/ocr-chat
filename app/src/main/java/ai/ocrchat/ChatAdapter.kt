package ai.ocrchat

import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

sealed class Msg {
    data class UserImage(val bmp: Bitmap, val name: String) : Msg()
    data class BotText(val text: String, val isErr: Boolean = false) : Msg()
    object BotTyping : Msg()
}

class ChatAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val items = ArrayList<Msg>()

    companion object {
        const val T_USER_IMG = 0
        const val T_BOT_TEXT = 1
        const val T_TYPING = 2
    }

    fun add(m: Msg) {
        items.add(m)
        notifyItemInserted(items.size - 1)
    }

    fun replaceTyping(m: Msg) {
        val i = items.indexOfLast { it is Msg.BotTyping }
        if (i >= 0) { items[i] = m; notifyItemChanged(i) } else add(m)
    }

    fun removeTyping() {
        val i = items.indexOfLast { it is Msg.BotTyping }
        if (i >= 0) { items.removeAt(i); notifyItemRemoved(i) }
    }

    override fun getItemViewType(position: Int) = when (items[position]) {
        is Msg.UserImage -> T_USER_IMG
        is Msg.BotText -> T_BOT_TEXT
        Msg.BotTyping -> T_TYPING
    }

    override fun getItemCount() = items.size

    class ImgH(v: View) : RecyclerView.ViewHolder(v) { val img: ImageView = v.findViewById(R.id.img) }
    class TextH(v: View) : RecyclerView.ViewHolder(v) { val tv: TextView = v.findViewById(R.id.msg) }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inf = LayoutInflater.from(parent.context)
        return when (viewType) {
            T_USER_IMG -> ImgH(inf.inflate(R.layout.item_msg_image, parent, false))
            else -> TextH(inf.inflate(R.layout.item_msg_text, parent, false))
        }
    }

    override fun onBindViewHolder(h: RecyclerView.ViewHolder, position: Int) {
        when (val m = items[position]) {
            is Msg.UserImage -> (h as ImgH).img.setImageBitmap(m.bmp)
            is Msg.BotText -> (h as TextH).tv.apply {
                text = m.text
                setTextColor(if (m.isErr) 0xFFFF9D9D.toInt() else 0xFFE6E9F0.toInt())
            }
            Msg.BotTyping -> (h as TextH).tv.apply {
                text = "⏳ распознаю…"
                setTextColor(0xFF8FA0C0.toInt())
            }
        }
    }
}
