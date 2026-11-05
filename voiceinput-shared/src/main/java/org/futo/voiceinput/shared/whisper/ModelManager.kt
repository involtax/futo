package org.futo.voiceinput.shared.whisper

import android.content.Context
import org.futo.voiceinput.shared.types.SpeechModel
import org.futo.voiceinput.shared.types.ModelLoader


class ModelManager(
    val context: Context
) {
    private val loadedModels: HashMap<Any, SpeechModel> = hashMapOf()

    @Synchronized
    fun obtainModel(model: ModelLoader): SpeechModel {
        val key = model.key(context)
        if (!loadedModels.contains(key)) {
            loadedModels[key] = model.loadModel(context)
        }

        return loadedModels[key]!!
    }

    @Synchronized
    fun cancelAll() {
        loadedModels.forEach {
            it.value.cancel()
        }
    }

    suspend fun cleanUp() {
        val models = synchronized(this) {
            loadedModels.values.toList().also { loadedModels.clear() }
        }
        for (model in models) {
            model.cancel()
            model.close()
        }
    }
}
