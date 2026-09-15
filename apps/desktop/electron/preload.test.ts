import assert from "node:assert/strict";
import { describe, it } from "vite-plus/test";
import { readableRejection } from "./ipc-errors.js";

// Pins the two string contracts readableRejection relies on: Electron wraps
// ipcMain.handle rejections as `Error invoking remote method '<channel>': …`
// and Effect TaggedErrors stringify as `<tag>: <message>` (Electron 44.3).
describe("readableRejection", () => {
  it("rewords transport failures after stripping the channel prefix", () => {
    assert.equal(
      readableRejection(
        "starling:transcribe",
        new Error(
          "Error invoking remote method 'starling:transcribe': RequestTransportError: fetch failed",
        ),
      ),
      "Could not reach the transcription server. fetch failed",
    );
  });

  it("keeps validation reasons verbatim without the class tag", () => {
    assert.equal(
      readableRejection(
        "starling:health",
        new Error(
          "Error invoking remote method 'starling:health': RequestInputError: Server endpoint must use http or https.",
        ),
      ),
      "Server endpoint must use http or https.",
    );
    assert.equal(
      readableRejection(
        "starling:health",
        new Error(
          "Error invoking remote method 'starling:health': RequestInputError: Put credentials in a trusted proxy, not the endpoint URL.",
        ),
      ),
      "Put credentials in a trusted proxy, not the endpoint URL.",
    );
  });

  it("strips timeout and http error tags", () => {
    assert.equal(
      readableRejection(
        "starling:health",
        new Error(
          "Error invoking remote method 'starling:health': RequestTimeoutError: Request timed out after 80 ms.",
        ),
      ),
      "Request timed out after 80 ms.",
    );
    assert.equal(
      readableRejection(
        "starling:transcribe",
        new Error(
          "Error invoking remote method 'starling:transcribe': RequestHttpError: Server returned 503: down",
        ),
      ),
      "Server returned 503: down",
    );
  });

  it("passes plain rejections through untouched", () => {
    assert.equal(readableRejection("starling:health", new Error("boom")), "boom");
    assert.equal(readableRejection("starling:health", "string cause"), "string cause");
  });
});
