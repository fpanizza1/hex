package ly.anoma.hex.keyboard

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import ly.anoma.hex.typing.KeyGrid
import ly.anoma.hex.typing.KeyPoint
import ly.anoma.hex.typing.SpatialDecoder
import ly.anoma.hex.typing.Suggestion
import ly.anoma.hex.typing.TypingModel
import ly.anoma.hex.typing.TypingWord

/** Labs › Keyboard › Second typing language, English and Spanish on the same keys: neither language's words are
 * "corrected" into the other, typos are fixed in the language the sentence is in, and the strip and next words
 * follow it. */
class BilingualTypingTest {
    private fun word(language: String, partner: String?): TypingWord {
        fun one(code: String) = TypingWord(SpatialDecoder(TypingModel.file(File("src/main/assets/typing/$code.hextm"), code)), spatial = true)
        return one(language).also { w -> w.partner = partner?.let(::one) }
    }

    private val grid = KeyGrid(buildMap {
        for ((row, letters) in listOf("qwertyuiop", "asdfghjkl", "zxcvbnm").withIndex())
            for ((i, c) in letters.withIndex()) put(c, KeyPoint(201 + (i - (letters.length - 1) / 2.0) * 39.5, 66.5 + 56 * row))
    }, 39.5, 56.0)

    private fun typed(language: String, partner: String?, before: String, text: String) = word(language, partner).also { w ->
        w.reset(before)
        for (c in text) grid.centres.getValue(c.lowercaseChar()).let { w.letter(c.toString(), it.x, it.y, grid) }
    }

    /** What space puts in place of [text] typed after [before], or null when it stays as typed. */
    private fun space(language: String, partner: String?, before: String, text: String) =
        typed(language, partner, before, text).finish(' ')?.replacement

    @Test fun eitherLanguagesWordsStayAsTyped() {
        // Alone, English "corrects" Spanish words.
        assertEquals("per", space("en", null, "", "pero"))
        assertEquals("goal", space("en", null, "", "hola"))
        for ((main, second) in listOf("en" to "es", "es" to "en")) {
            for (spanish in listOf("pero", "hola", "que", "gracias")) assertNull("$main+$second $spanish", space(main, second, "", spanish))
            assertNull("$main+$second cosa", space(main, second, "the ", "cosa"))
        }
    }

    @Test fun typosAreFixedInTheSentencesLanguage() {
        for ((main, second) in listOf("en" to "es", "es" to "en")) {
            assertEquals("que", space(main, second, "hola como ", "qie"))
            assertEquals("mañana", space(main, second, "nos vemos ", "manana"))
            assertEquals("the", space(main, second, "I think ", "teh"))
            assertEquals("don't", space(main, second, "see you ", "dont"))
            assertEquals("tomorrow", space(main, second, "the meeting is ", "tomorow"))
        }
        // Spanish books spell "dont" and "im", which never keeps English's own fix of them.
        assertEquals("don't", space("en", "es", "I ", "dont"))
        assertEquals("I'm", space("en", "es", "", "im"))
    }

    @Test fun theStripAndNextWordsFollowTheSentence() {
        val strip = typed("en", "es", "hola como ", "est").strip().map { it.text }
        assertTrue(strip.toString(), "está" in strip || "este" in strip)
        // A word the other language knows keeps its literal slot.
        assertEquals(Suggestion("cosa", Suggestion.Kind.Typed), typed("en", "es", "the ", "cosa").strip().first())
        val spanishNext = word("en", "es").predictions("hola como estas ")
        val englishNext = word("en", "es").predictions("how are you ")
        assertTrue(spanishNext.toString(), spanishNext.isNotEmpty() && spanishNext.none { it in englishNext })
    }
}
