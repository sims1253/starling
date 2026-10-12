package dev.starling.mobile.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Every catalog download is pinned, verifiable and uniquely named. */
class ModelCatalogTest {
    private val pinned = Regex("""^https://huggingface\.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/[^/]+\.gguf$""")

    @Test
    fun everyEntryPinsARevisionAndItsBytes() {
        for (spec in ModelCatalog.ALL) {
            assertTrue("not pinned to a revision: ${spec.url}", pinned.matches(spec.url))
            assertTrue("bad SHA-256 for ${spec.fileName}", Regex("^[0-9a-f]{64}$").matches(spec.sha256))
            assertTrue("no size for ${spec.fileName}", spec.sizeBytes > 0)
        }
    }

    @Test
    fun fileNamesAndLabelsAreUnique() {
        val all = ModelCatalog.ALL
        assertEquals(all.size, all.map { it.fileName }.toSet().size)
        assertEquals(all.size, all.map { it.label }.toSet().size)
    }

    @Test
    fun theRecommendedModelComesFirst() {
        assertEquals(ModelCatalog.RECOMMENDED_PARAKEET, ModelCatalog.ALL.first())
        for (spec in ModelCatalog.ALL) assertEquals(spec, ModelCatalog.forFileName(spec.fileName))
    }
}
