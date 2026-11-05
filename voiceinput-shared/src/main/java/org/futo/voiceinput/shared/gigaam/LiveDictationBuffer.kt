package org.futo.voiceinput.shared.gigaam

/** Audio is retained until decoded. There is only one in-flight snapshot, never a job per frame. */
class LiveDictationBuffer {
    data class Snapshot(val samples: FloatArray, val finalize: Boolean, val rollover: Boolean)
    private var audio = FloatArray(16000 * 16)
    private var size = 0
    private var lastSpeech = -1
    private var lastPreviewSize = 0
    private var committed = ""
    private var preview = ""
    private var overlapSamples = 0
    @get:Synchronized val text: String get() = join(committed, preview)

    @Synchronized
    fun append(samples: FloatArray, speech: Boolean) {
        if (size + samples.size > audio.size) audio = audio.copyOf(maxOf(audio.size * 2, size + samples.size))
        samples.copyInto(audio, size)
        size += samples.size
        if (speech) lastSpeech = size
        // Ignore unbounded initial silence (or silence between completed phrases).
        if (lastSpeech < 0 && size > 8000) {
            audio.copyInto(audio, 0, size - 4800, size)
            size = 4800
        }
    }

    @Synchronized
    fun snapshot(finishing: Boolean): Snapshot? {
        if (lastSpeech < 0 || size < 1600) return null
        val count = minOf(size, 16000 * 12)
        val pause = lastSpeech <= count - 11200
        if (!finishing && !pause && size - lastPreviewSize < 16000) return null
        lastPreviewSize = size
        return Snapshot(audio.copyOf(count), pause || (finishing && count == size), count == 16000 * 12)
    }

    @Synchronized
    fun accept(snapshot: Snapshot, result: DecodedSpeech) {
        val fresh = withoutCommittedOverlap(result)
        if (snapshot.finalize) {
            committed = join(committed, fresh.text)
            preview = ""
            consume(snapshot.samples.size)
            overlapSamples = 0
        } else if (snapshot.rollover) {
            // Cut at a decoded word boundary, keeping ~2 seconds uncommitted for context.
            // Decoder tokens have explicit word separators, with timestamps relative to this window.
            val boundary = fresh.tokens.indices.lastOrNull {
                it < fresh.timestamps.size && fresh.tokens[it].isBlank() &&
                    fresh.timestamps[it] in 1.0f..10.0f
            }
            if (boundary != null) {
                val prefix = fresh.tokens.take(boundary).joinToString("").trim()
                committed = join(committed, prefix)
                preview = fresh.tokens.drop(boundary + 1).joinToString("").trim()
                // CTC timestamps mark emitted frames, not the exact acoustic word end.
                // Cut inside the separating space, before the next emitted character.
                val next = fresh.timestamps.getOrNull(boundary + 1) ?: fresh.timestamps[boundary]
                val end = (fresh.timestamps[boundary] + next) / 2f
                // Retain up to two seconds of already decoded words as acoustic context.
                // Only remove matching words when that context is actually present.
                val context = result.tokens.indices.firstOrNull {
                    it < result.timestamps.size && result.tokens[it].isBlank() &&
                        result.timestamps[it] >= end - 2f && result.timestamps[it] < fresh.timestamps[boundary] - 0.1f
                }
                val start = if (context != null) result.timestamps[context] else end
                val count = (start * 16000).toInt().coerceIn(1, snapshot.samples.size)
                consume(count)
                overlapSamples = (end * 16000).toInt() - count
            } else {
                // A pathological window without word boundaries still makes forward progress.
                committed = join(committed, fresh.text)
                preview = ""
                consume(snapshot.samples.size)
                overlapSamples = 0
            }
        } else {
            preview = fresh.text
        }
    }

    private fun withoutCommittedOverlap(result: DecodedSpeech): DecodedSpeech {
        if (overlapSamples <= 0) return result
        val raw = result.tokens.joinToString("")
        val words = Regex("\\S+").findAll(raw).toList()
        val oldWords = committed.split(Regex("\\s+"))
        val matching = (minOf(12, words.size, oldWords.size) downTo 1).firstOrNull { n ->
            words.take(n).map { normalizeWord(it.value) } == oldWords.takeLast(n).map(::normalizeWord)
        }
        val drop = if (matching != null) {
            val chars = words[matching - 1].range.last + 1
            var count = 0
            result.tokens.indexOfFirst { count += it.length; count >= chars } + 1
        } else {
            // If the acoustic context was transcribed differently, use the CTC alignment.
            val seconds = overlapSamples / 16000f
            (result.tokens.indices.firstOrNull {
                it < result.timestamps.size && result.tokens[it].isBlank() && result.timestamps[it] >= seconds
            } ?: -1) + 1
        }
        val tokens = result.tokens.drop(drop)
        return DecodedSpeech(tokens.joinToString("").trim(), tokens, result.timestamps.drop(drop))
    }

    private fun consume(count: Int) {
        audio.copyInto(audio, 0, count, size)
        size -= count
        lastSpeech = if (lastSpeech > count) lastSpeech - count else -1
        lastPreviewSize = 0
    }

    private fun join(a: String, b: String) = listOf(a.trim(), b.trim()).filter { it.isNotEmpty() }.joinToString(" ")

    private fun normalizeWord(word: String) = word.lowercase().replace('ё', 'е').replace(Regex("[^\\p{L}\\p{N}]"), "")
}
