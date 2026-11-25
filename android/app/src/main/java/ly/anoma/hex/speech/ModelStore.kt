package ly.anoma.hex.speech

import android.app.Activity
import android.app.Application
import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Bundle
import kotlinx.coroutines.flow.MutableStateFlow
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

/** Whether the phone's network costs nothing per byte, as Android judges it: true on most Wi-Fi, false on mobile data or a
 * metered Wi-Fi (a phone's hotspot), null with no network. Android leaves the network's name and other location details
 * out of what it returns here; this is all Hex asks about the network (ACCESS_NETWORK_STATE). */
internal fun unmeteredNetwork(context: Context): Boolean? {
    val connectivity = context.getSystemService(ConnectivityManager::class.java) ?: return null
    return connectivity.getNetworkCapabilities(connectivity.activeNetwork ?: return null)?.let(::unmetered)
}
private fun unmetered(capabilities: NetworkCapabilities) = capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) ||
    capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_TEMPORARILY_NOT_METERED)

/** A Wi-Fi-first download waits on a network that charges for data, unless the person chose mobile data for it. With
 * no network at all it goes ahead and fails as before, saying there's no connection. */
internal fun waitsForWifi(mobileDataChosen: Boolean, unmetered: Boolean?) = !mobileDataChosen && unmetered == false

/** A Hex screen in front of another app that must make no network request: a model download in progress does not
 * resume while it shows (Labs › Dictate from Quick Settings' sheet). */
internal interface OfflineScreen

/** Hex's network operation: an explicit, fixed model GET from Hex's own screens, each file pinned by revision, size and
 * SHA-256 ([small] from setup or Dictation settings, [english] from Labs or, with Labs › Pick the English model for this
 * phone, when Hex is opened). The keyboard never calls it. A wifiFirst download (both models, whoever asked) waits for
 * an unmetered network ([State.WaitingForWifi]) unless the person chooses mobile data. */
internal class ModelStore private constructor(private val name: String, private val size: Long, private val hash: String,
                                              private val url: String, private val extension: String,
                                              internal val wifiFirst: Boolean = false) {
    companion object {
        /** Whisper small, the default speech model. */
        val small = ModelStore("whisper-small", 190085487L, "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q5_1.bin", "bin",
            wifiFirst = true)
        /** Labs › Parakeet: NVIDIA's Parakeet TDT 0.6B v3 (CC BY 4.0) in k2-fsa's int8 ONNX conversion, the files
         * Android's benchmark measured, at a fixed revision. Four files, so a [PinnedModel]. */
        val english = PinnedModel("parakeet-tdt-0.6b-v3-int8-ed5d7456", listOf(
            "encoder.int8.onnx" to (652282294L to "1afbe69c1ffdabee94755f42e4daf1d93549131a3da11356bda84fcef0f2c479"),
            "decoder.int8.onnx" to (11845274L to "0222a86147299d4dee90edf95b6d13827edb42ad8b7c9cf0b2582f030937c0b5"),
            "joiner.int8.onnx" to (6355277L to "cbd02f3696e9944970111fb439d9c3fd365a80374565f13f2b9c7007d253fc98"),
            "tokens.txt" to (93939L to "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
        ).map { (name, pin) ->
            Pinned(name, pin.first, pin.second,
                "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/" +
                    "ed5d745623472f97cfc2e6c2916ee248798fc5a8/$name")
        })
    }

    /** One file of a multi-file model: its fixed HTTPS address, exact size, SHA-256 and its name on the phone. */
    class Pinned(val name: String, val size: Long, val hash: String, val url: String)


    enum class Problem { Connection, Server, Space }
    sealed interface State {
        data object Available : State
        data object Ready : State
        data class Retry(val problem: Problem) : State
        data class Loading(val percent: Int) : State
        /** On a network that charges for data: it starts by itself on Wi-Fi, or now with [useMobileData]. */
        data object WaitingForWifi : State
    }
    private class Failure(val problem: Problem) : Exception()
    private class Deleted : Exception()
    private val progress = MutableStateFlow<State>(State.Available)
    val state = progress.asStateFlow()
    private var running = false
    private var waiting = false
    private var watching = false
    private var started = 0
    private var stops = 0
    /** Bumped by [delete], so a transfer in progress stops and leaves nothing behind. */
    @Volatile private var generation = 0
    /** The installed file's name, after its SHA-256. */
    internal val fileName get() = "$name-$hash.$extension"
    private fun path(context: Context) = File(context.noBackupFilesDir, fileName)
    private fun part(context: Context) = File(context.noBackupFilesDir, "$name-$hash.part")
    fun installed(context: Context) = installedIn(context.noBackupFilesDir)
    internal fun installedIn(dir: File) = File(dir, fileName).takeIf { it.isFile && it.length() == size }
    /** The download's size, said before it starts. */
    val megabytes get() = ((size + 500_000) / 1_000_000).toInt()
    /** The file checked against its SHA-256 in this process, so it is hashed once, not at every load. */
    @Volatile private var checked: File? = null

    /** [installed], once its contents match the pinned SHA-256 (hashed the first time in each process, on the caller's
     * thread). A file that doesn't is deleted, so its row offers the download again. */
    fun verified(context: Context) = verifiedIn(context.noBackupFilesDir)
    internal fun verifiedIn(dir: File): File? {
        val file = installedIn(dir) ?: return null
        if (checked == file) return file
        val digest = MessageDigest.getInstance("SHA-256")
        val ok = runCatching { file.inputStream().use { input -> pump(input) { buffer, count -> digest.update(buffer, 0, count) } } }.isSuccess &&
            digest.digest().joinToString("") { "%02x".format(it) } == hash
        if (ok) { checked = file; return file }
        synchronized(this) { file.delete(); if (!running) progress.value = State.Available }
        return null
    }

    /** Removes the file and any partial download, stopping a transfer in progress; [state] is Available again. */
    @Synchronized fun delete(context: Context) {
        val app = context.applicationContext
        generation++
        waiting = false
        checked = null
        path(app).delete(); part(app).delete()
        forgetMobileData(app)
        progress.value = State.Available
    }

    /** The person's choice to download on mobile data, kept until this file is installed or deleted, so a download that
     * resumes later doesn't wait for Wi-Fi again. */
    private fun choices(app: Context) = app.getSharedPreferences("models", Context.MODE_PRIVATE)
    private fun mobileDataChosen(app: Context) = choices(app).getBoolean("$name.mobileData", false)
    private fun forgetMobileData(app: Context) = choices(app).edit().remove("$name.mobileData").apply()

    /** Downloads now on mobile data too (the waiting row's action). */
    @Synchronized fun useMobileData(context: Context) {
        val app = context.applicationContext
        choices(app).edit().putBoolean("$name.mobileData", true).apply()
        download(app)
    }

    private var network: ConnectivityManager.NetworkCallback? = null
    /** While it waits for Wi-Fi, an unmetered network starts it if a Hex screen is in front (Android closes a backgrounded
     * app's connections; a screen coming to the front checks again). */
    private fun watchNetwork(app: Context) {
        if (network != null) return
        val connectivity = app.getSystemService(ConnectivityManager::class.java) ?: return
        network = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(changed: Network, capabilities: NetworkCapabilities) {
                if (unmetered(capabilities)) synchronized(this@ModelStore) {
                    if (progress.value == State.WaitingForWifi && started > 0) download(app)
                }
            }
        }.also { connectivity.registerDefaultNetworkCallback(it) }
    }

    @Synchronized fun refresh(context: Context) {
        val app = context.applicationContext as Application
        if (!watching) { watching = true; app.registerActivityLifecycleCallbacks(foreground) }
        if (installed(app) != null) progress.value = State.Ready
        else if (progress.value == State.Ready) progress.value = State.Available
    }

    // Android closes a backgrounded app's connections, and a swipe from recents or a reboot ends the process.
    // A started download keeps its partial file and carries on whenever a Hex screen is in front again.
    private val foreground = object : Application.ActivityLifecycleCallbacks {
        override fun onActivityStarted(activity: Activity) {
            if (activity is OfflineScreen) return
            synchronized(this@ModelStore) {
                if (started++ == 0 && !running && (waiting || part(activity).exists()) && installed(activity) == null) download(activity)
            }
        }
        override fun onActivityStopped(activity: Activity) {
            if (activity is OfflineScreen) return
            synchronized(this@ModelStore) { if (--started == 0) stops++ }
        }
        override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
        override fun onActivityResumed(activity: Activity) = Unit
        override fun onActivityPaused(activity: Activity) = Unit
        override fun onActivitySaveInstanceState(activity: Activity, state: Bundle) = Unit
        override fun onActivityDestroyed(activity: Activity) = Unit
    }

    /** Whether a download asked for now starts at once: always for a file that isn't Wi-Fi-first, else unless the network
     * charges for data ([unmetered] false) and the person didn't choose mobile data. A tap and Labs › Pick the English model
     * for this phone ask the same way ([download]); only Use mobile data ([useMobileData]) chooses it. */
    internal fun startsNow(mobileDataChosen: Boolean, unmetered: Boolean?) = !wifiFirst || !waitsForWifi(mobileDataChosen, unmetered)

    @Synchronized fun download(context: Context) {
        val app = context.applicationContext
        if (installed(app) != null) { progress.value = State.Ready; return }
        if (running) return
        if (!startsNow(mobileDataChosen(app), unmeteredNetwork(app))) {
            waiting = true; progress.value = State.WaitingForWifi; watchNetwork(app); return
        }
        running = true; waiting = false
        val part = part(app)
        progress.value = State.Loading((part.length().coerceIn(0, size) * 100 / size).toInt().coerceAtMost(99))
        val since = stops
        val mine = generation
        Thread({
            val result = try {
                val resumed = part.length() > 0
                // A resumed file that fails the checksum is discarded and fetched once more from the start.
                if (transfer(app, part, mine) || resumed && transfer(app, part, mine)) State.Ready else State.Retry(Problem.Server)
            } catch (_: Deleted) { null }
            catch (e: Failure) { State.Retry(e.problem) }
            catch (_: IOException) { State.Retry(if (app.noBackupFilesDir.usableSpace < 8_000_000) Problem.Space else Problem.Connection) }
            catch (_: Exception) { State.Retry(Problem.Server) }
            synchronized(this) {
                running = false
                if (result == null || generation != mine) {
                    // Deleted while it ran: whatever the transfer wrote last goes too.
                    part.delete(); path(app).delete(); progress.value = State.Available
                // A Wi-Fi-first download whose Wi-Fi went away waits for it again rather than saying there's no connection.
                } else if (result == State.Retry(Problem.Connection) && (started == 0 || stops != since ||
                        !startsNow(mobileDataChosen(app), unmeteredNetwork(app)))) {
                    waiting = true
                    if (started > 0) download(app)
                } else progress.value = result
            }
        }, "Hex $name download").start()
    }

    /** Returns false, after deleting the partial file, when the finished file fails the checksum. */
    private fun transfer(app: Context, part: File, mine: Int): Boolean {
        if (part.length() > size) part.delete()
        if (app.noBackupFilesDir.usableSpace <= size - part.length() + 8_000_000) throw Failure(Problem.Space)
        val digest = MessageDigest.getInstance("SHA-256")
        var total = part.length()
        if (total > 0) part.inputStream().use { input -> pump(input) { buffer, count -> digest.update(buffer, 0, count) } }
        var http = open(total)
        if (total > 0 && if (http == null) total < size else http.responseCode == 206 && !resumes(http, total)) {
            http?.disconnect(); part.delete(); total = 0; digest.reset()
            http = open(0) ?: throw Failure(Problem.Server)
        }
        if (http != null) try {
            if (http.responseCode == 200) { total = 0; digest.reset() }
            http.inputStream.use { input ->
                FileOutputStream(part, total > 0).use { output ->
                    pump(input) { buffer, count ->
                        if (generation != mine) throw Deleted()
                        total += count
                        if (total > size) throw Failure(Problem.Server)
                        digest.update(buffer, 0, count); output.write(buffer, 0, count)
                        progress.value = State.Loading((total * 100 / size).toInt().coerceAtMost(99))
                    }
                    output.fd.sync()
                }
            }
        } finally { http.disconnect() }
        if (total < size) throw IOException("Model download ended early")
        if (digest.digest().joinToString("") { "%02x".format(it) } != hash) { part.delete(); return false }
        synchronized(this) {
            if (generation != mine) throw Deleted()
            Files.move(part.toPath(), path(app).toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            forgetMobileData(app)
        }
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

    private fun resumes(http: HttpsURLConnection, from: Long) =
        http.getHeaderField("Content-Range") == "bytes $from-${size - 1}/$size"

    /** Opens the model at [from] bytes; null when the server says nothing is left to send. */
    private fun open(from: Long): HttpsURLConnection? {
        var target = URL(url)
        repeat(6) {
            val host = target.host.lowercase()
            if (target.protocol != "https" || target.userInfo != null || target.port !in listOf(-1, 443) ||
                !(host == "huggingface.co" || host.endsWith(".cdn.hf.co") || host == "cas-bridge.xethub.hf.co")) throw Failure(Problem.Server)
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
                    301, 302, 303, 307, 308 -> target = URL(target, http.getHeaderField("Location") ?: throw Failure(Problem.Server))
                    else -> throw Failure(Problem.Server)
                }
            } catch (e: Exception) { http.disconnect(); throw e }
            http.disconnect()
        }
        throw Failure(Problem.Server)
    }
}

/** A model of several pinned files ([folder] under no-backup storage), fetched one after another. */
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
    /** The download's size, said before it starts. */
    val megabytes get() = ((total + 500_000) / 1_000_000).toInt()
    private fun dir(context: Context) = File(context.noBackupFilesDir, folder)
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
        if (running) return
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
