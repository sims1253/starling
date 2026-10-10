package dev.starling.mobile.processing

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class ContractsSmokeTest {
    @Test
    fun contractTablesAreReachable() {
        assertEquals(1, JSONObject(Contracts.modeRouting("spoken-commands.json")).getInt("schema_version"))
    }
}
