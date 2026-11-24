package ly.anoma.hex.speech

import android.content.Context
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig

/** Labs › Dictation › Parakeet: NVIDIA's Parakeet TDT 0.6B v3 through sherpa-onnx, off by default. It is about six
 * times faster than Whisper small on short clips and needs about 1.4 GB of memory while loaded. Whisper still
 * dictates whatever it can't: Norwegian, translation, and any load or decode failure. Called under [SpeechModel]'s
 * lock only. */
internal object Parakeet {
    /** The 25 European languages the model card lists. */
    private val languages = setOf("bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
        "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk")
    private var recognizer: OfflineRecognizer? = null
    /** A load that failed for these files is not retried until they change or the process restarts. */
    private var failed: List<String>? = null

    fun wanted(app: Context, language: String, translate: Boolean) =
        SpeechPreferences(app).parakeet && !translate && language in languages && ModelStore.parakeet.installed(app) != null

    fun load(app: Context, threads: Int): OfflineRecognizer? {
        recognizer?.let { return it }
        val files = ModelStore.parakeet.installed(app)?.map { it.absolutePath } ?: return null
        if (files == failed) return null
        val (encoder, decoder, joiner, tokens) = files
        recognizer = try {
            OfflineRecognizer(null, OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
                modelConfig = OfflineModelConfig(
                    transducer = OfflineTransducerModelConfig(encoder = encoder, decoder = decoder, joiner = joiner),
                    tokens = tokens, numThreads = threads, modelType = "nemo_transducer", debug = false),
                decodingMethod = "greedy_search"))
        } catch (_: Throwable) { failed = files; null }
        return recognizer
    }

    fun decode(recognizer: OfflineRecognizer, pcm: ShortArray, count: Int): String {
        val samples = FloatArray(count) { pcm[it] / 32768f }
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(samples, 16000)
            recognizer.decode(stream)
            return recognizer.getResult(stream).text
        } finally { stream.release() }
    }

    fun unload() { recognizer?.release(); recognizer = null }
}
