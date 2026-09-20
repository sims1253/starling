// Request*Error messages reach the UI verbatim; only transport failures need
// rewording, and the remaining internal error-class tags are stripped.
export function readableRejection(channel: string, cause: unknown): string {
  const channelPrefix = `Error invoking remote method '${channel}': `;
  const raw = cause instanceof Error ? cause.message : String(cause);
  let message = raw.startsWith(channelPrefix) ? raw.slice(channelPrefix.length) : raw;

  if (message.startsWith("RequestTransportError: ")) {
    return `Could not reach the transcription server. ${message.slice("RequestTransportError: ".length)}`;
  }

  message = message.replace(/^(?:Request(?:Input|Timeout|Http)Error):\s*/, "");

  return message;
}
