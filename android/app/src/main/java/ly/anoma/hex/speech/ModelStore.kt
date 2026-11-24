package ly.anoma.hex.speech

import android.app.Activity
import android.app.Application
import android.content.Context
import android.os.Bundle
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import javax.net.ssl.HttpsURLConnection

/** The only network operations in Hex: explicit, fixed model GETs from setup, Dictation settings or Labs. */
internal object ModelStore {
    enum class Problem { Connection, Server, Space }
    sealed interface State {
        data object Available : State
        data object Ready : State
        data class Retry(val problem: Problem) : State
        data class Loading(val percent: Int) : State
    }

    /** One file of a model: its fixed HTTPS address, exact size and SHA-256, and its name on the phone. */
    class Pinned(val url: String, val size: Long, val hash: String, val name: String)

    /** Whisper small Q5_1, the default model. */
    private val small = PinnedModel("", listOf(Pinned(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin",
        190085487L, "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        "whisper-small-ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb.bin")))

    /** Labs › Dictation › Parakeet: NVIDIA's Parakeet TDT 0.6B v3 (CC BY 4.0) in k2-fsa's int8 ONNX conversion,
     * the files Android's benchmark measured, at a fixed revision. */
    val parakeet = PinnedModel("parakeet-tdt-0.6b-v3-int8-ed5d7456", listOf(
        "encoder.int8.onnx" to (652282294L to "1afbe69c1ffdabee94755f42e4daf1d93549131a3da11356bda84fcef0f2c479"),
        "decoder.int8.onnx" to (11845274L to "0222a86147299d4dee90edf95b6d13827edb42ad8b7c9cf0b2582f030937c0b5"),
        "joiner.int8.onnx" to (6355277L to "cbd02f3696e9944970111fb439d9c3fd365a80374565f13f2b9c7007d253fc98"),
        "tokens.txt" to (93939L to "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
    ).map { (name, pin) ->
        Pinned("https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/" +
            "ed5d745623472f97cfc2e6c2916ee248798fc5a8/$name", pin.first, pin.second, name)
    })

    val state: StateFlow<State> get() = small.state
    fun installed(context: Context): File? = small.installed(context)?.single()
    fun refresh(context: Context) { small.refresh(context); parakeet.refresh(context) }
    fun download(context: Context) = small.download(context)
}

/** A model's pinned files in no-backup storage ([folder] under it, "" for none), fetched one after another. */
internal class PinnedModel(private val folder: String, private val files: List<ModelStore.Pinned>) {
    private val total = files.sumOf { it.size }
    private class Failure(val problem: ModelStore.Problem) : Exception()
    private val progress = MutableStateFlow<ModelStore.State>(ModelStore.State.Available)
    val state = progress.asStateFlow()
    private var running = false
    private var waiting = false
    private var watching = false
    private var started = 0
    private var stops = 0
    private fun dir(context: Context) = if (folder.isEmpty()) context.noBackupFilesDir else File(context.noBackupFilesDir, folder)
    private fun path(context: Context, file: ModelStore.Pinned) = File(dir(context), file.name)
    private fun part(context: Context, file: ModelStore.Pinned) = File(dir(context), file.name.substringBeforeLast('.') + ".part")
    private fun done(context: Context, file: ModelStore.Pinned) = path(context, file).let { it.isFile && it.length() == file.size }
    private fun anyPart(context: Context) = files.any { part(context, it).exists() }
    /** Every file, in order, once all are on the phone; else null. */
    fun installed(context: Context): List<File>? = if (files.all { done(context, it) }) files.map { path(context, it) } else null

    @Synchronized fun refresh(context: Context) {
        val app = context.applicationContext as Application
        if (!watching) { watching = true; app.registerActivityLifecycleCallbacks(foreground) }
        if (installed(app) != null) progress.value = ModelStore.State.Ready
    }

    // Android closes a backgrounded app's connections, and a swipe from recents or a reboot ends the process.
    // A started download keeps its partial file and carries on whenever a Hex screen is in front again.
    private val foreground = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityStarted(activity: Activity) {
            synchronized(this@PinnedModel) {
                if (started++ == 0 && !running && (waiting || anyPart(activity)) && installed(activity) == null) download(activity)
            }
        }
        override fun onActivityStopped(activity: Activity) { synchronized(this@PinnedModel) { if (--started == 0) stops++ } }
        override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
        override fun onActivityResumed(activity: Activity) = Unit
        override fun onActivityPaused(activity: Activity) = Unit
        override fun onActivitySaveInstanceState(activity: Activity, state: Bundle) = Unit
        override fun onActivityDestroyed(activity: Activity) = Unit
    }

    /** Removes the files and any partial ones; nothing while a download runs. */
    @Synchronized fun delete(context: Context) {
        val app = context.applicationContext
        if (running || folder.isEmpty()) return
        waiting = false
        dir(app).deleteRecursively()
        progress.value = ModelStore.State.Available
    }

    /** Bytes already on the phone: finished files and partial ones. */
    private fun have(app: Context) = files.sumOf { if (done(app, it)) it.size else part(app, it).length().coerceIn(0, it.size) }
    private fun percent(bytes: Long) = (bytes * 100 / total).toInt().coerceAtMost(99)

    @Synchronized fun download(context: Context) {
        val app = context.applicationContext
        if (installed(app) != null) { progress.value = ModelStore.State.Ready; return }
        if (running) return
        running = true; waiting = false
        dir(app).mkdirs()
        progress.value = ModelStore.State.Loading(percent(have(app)))
        val since = stops
        Thread({
            val result = try {
                var ok = true
                for (file in files) {
                    if (done(app, file)) continue
                    val part = part(app, file)
                    val resumed = part.length() > 0
                    // A resumed file that fails the checksum is discarded and fetched once more from the start.
                    if (!(transfer(app, file, part) || resumed && transfer(app, file, part))) { ok = false; break }
                }
                if (ok) ModelStore.State.Ready else ModelStore.State.Retry(ModelStore.Problem.Server)
            } catch (e: Failure) { ModelStore.State.Retry(e.problem) }
            catch (_: IOException) {
                ModelStore.State.Retry(if (app.noBackupFilesDir.usableSpace < 8_000_000) ModelStore.Problem.Space else ModelStore.Problem.Connection)
            }
            catch (_: Exception) { ModelStore.State.Retry(ModelStore.Problem.Server) }
            synchronized(this) {
                running = false
                if (result == ModelStore.State.Retry(ModelStore.Problem.Connection) && (started == 0 || stops != since)) {
                    waiting = true
                    if (started > 0) download(app)
                } else progress.value = result
            }
        }, "Hex model download").start()
    }

    /** Returns false, after deleting the partial file, when the finished file fails the checksum. */
    private fun transfer(app: Context, file: ModelStore.Pinned, part: File): Boolean {
        val size = file.size
        if (part.length() > size) part.delete()
        val left = files.sumOf { if (done(app, it)) 0L else it.size - part(app, it).length().coerceIn(0, it.size) }
        if (app.noBackupFilesDir.usableSpace <= left + 8_000_000) throw Failure(ModelStore.Problem.Space)
        val before = have(app) - part.length()
        val digest = MessageDigest.getInstance("SHA-256")
        var count = part.length()
        if (count > 0) part.inputStream().use { input -> pump(input) { buffer, n -> digest.update(buffer, 0, n) } }
        var http = open(file, count)
        if (count > 0 && if (http == null) count < size else http.responseCode == 206 && !resumes(http, count, size)) {
            http?.disconnect(); part.delete(); count = 0; digest.reset()
            http = open(file, 0) ?: throw Failure(ModelStore.Problem.Server)
        }
        if (http != null) try {
            if (http.responseCode == 200) { count = 0; digest.reset() }
            http.inputStream.use { input ->
                FileOutputStream(part, count > 0).use { output ->
                    pump(input) { buffer, n ->
                        count += n
                        if (count > size) throw Failure(ModelStore.Problem.Server)
                        digest.update(buffer, 0, n); output.write(buffer, 0, n)
                        progress.value = ModelStore.State.Loading(percent(before + count))
                    }
                    output.fd.sync()
                }
            }
        } finally { http.disconnect() }
        if (count < size) throw IOException("Model download ended early")
        if (digest.digest().joinToString("") { "%02x".format(it) } != file.hash) { part.delete(); return false }
        Files.move(part.toPath(), path(app, file).toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        return true
    }

    private inline fun pump(input: InputStream, each: (ByteArray, Int) -> Unit) {
        val buffer = ByteArray(128 * 1024)
        while (true) {
            val count = input.read(buffer)
            if (count == -1) break
            each(buffer, count)
        }
    }

    private fun resumes(http: HttpsURLConnection, from: Long, size: Long) =
        http.getHeaderField("Content-Range") == "bytes $from-${size - 1}/$size"

    /** Opens [file] at [from] bytes; null when the server says nothing is left to send. */
    private fun open(file: ModelStore.Pinned, from: Long): HttpsURLConnection? {
        var target = URL(file.url)
        repeat(6) {
            val host = target.host.lowercase()
            if (target.protocol != "https" || target.userInfo != null || target.port !in listOf(-1, 443) ||
                !(host == "huggingface.co" || host.endsWith(".cdn.hf.co") || host == "cas-bridge.xethub.hf.co")) throw Failure(ModelStore.Problem.Server)
            val http = target.openConnection() as HttpsURLConnection
            try {
                http.instanceFollowRedirects = false
                http.connectTimeout = 15_000; http.readTimeout = 45_000
                // Android's default user agent includes the phone model and OS build.
                http.setRequestProperty("User-Agent", "HexModelDownloader/1")
                http.setRequestProperty("Accept-Encoding", "identity")
                if (from > 0) http.setRequestProperty("Range", "bytes=$from-")
                when (http.responseCode) {
                    200, 206 -> return http
                    416 -> { http.disconnect(); return null }
                    301, 302, 303, 307, 308 -> target = URL(target, http.getHeaderField("Location") ?: throw Failure(ModelStore.Problem.Server))
                    else -> throw Failure(ModelStore.Problem.Server)
                }
            } catch (e: Exception) { http.disconnect(); throw e }
            http.disconnect()
        }
        throw Failure(ModelStore.Problem.Server)
    }
}
