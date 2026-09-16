import StarlingVoiceCore
import SwiftUI
import UIKit

struct SessionDetailView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let session: SessionRecord
    let configuration: ServerConfiguration
    @ObservedObject var playback: AudioPlayback
    @State private var confirmsDelete = false

    private var current: SessionRecord {
        model.sessions.first { $0.id == session.id } ?? session
    }

    private var analysis: FidelityAnalysis? {
        guard let transcript = current.transcript else { return nil }
        let recordingSeconds = current.durationMilliseconds.map { Double($0) / 1_000 }
        let covered = transcript.segments.map(\.endSeconds).max()
        return FidelityAnalyzer.analyze(
            transcript.text,
            recordingDurationSeconds: recordingSeconds,
            coveredDurationSeconds: covered
        )
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    statusCard
                    if let transcript = current.transcript {
                        transcriptCard(transcript)
                    }
                    if !current.transcriptHistory.isEmpty {
                        transcriptHistoryCard(current.transcriptHistory)
                    }
                    if let analysis, !analysis.warnings.isEmpty {
                        warningsCard(analysis.warnings)
                    }
                    actions
                }
                .padding(20)
            }
            .background(StarlingTheme.charcoal.ignoresSafeArea())
            .navigationTitle(current.createdAt.formatted(date: .abbreviated, time: .shortened))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .confirmationDialog(
                "Delete this recording and transcript?",
                isPresented: $confirmsDelete,
                titleVisibility: .visible
            ) {
                Button("Delete permanently", role: .destructive) {
                    Task { await model.delete(current) }
                }
            }
        }
    }

    private var statusCard: some View {
        HStack(spacing: 12) {
            Image(systemName: current.status == .failed ? "exclamationmark.circle.fill" : "waveform.circle.fill")
                .font(.title2)
                .foregroundStyle(current.status == .failed ? StarlingTheme.coral : StarlingTheme.lime)
            VStack(alignment: .leading, spacing: 3) {
                Text(current.status.rawValue.capitalized)
                    .font(.headline)
                Text("\(durationText) · \(current.attemptCount) request\(current.attemptCount == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 18))
    }

    private func transcriptCard(_ transcript: Transcript) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Raw transcript")
                    .font(.headline)
                Spacer()
                Button {
                    UIPasteboard.general.string = transcript.text
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                        .labelStyle(.iconOnly)
                }
                ShareLink(item: transcript.text) {
                    Image(systemName: "square.and.arrow.up")
                }
            }
            Text(transcript.text.isEmpty ? "The server returned an empty transcript." : transcript.text)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .background(StarlingTheme.paper, in: RoundedRectangle(cornerRadius: 20))
        .foregroundStyle(StarlingTheme.ink)
    }

    private func warningsCard(_ warnings: [FidelityWarning]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Review cues", systemImage: "eye")
                .font(.headline)
            ForEach(warnings) { warning in
                Text(warning.message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Text("These cues cannot determine whether recognition was correct. Compare with the saved audio.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(18)
        .background(StarlingTheme.amber.opacity(0.08), in: RoundedRectangle(cornerRadius: 20))
    }

    private func transcriptHistoryCard(_ transcripts: [Transcript]) -> some View {
        DisclosureGroup("Earlier retry results (\(transcripts.count))") {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(Array(transcripts.enumerated()), id: \.offset) { index, transcript in
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Attempt \(index + 1)")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(transcript.text.isEmpty ? "Empty transcript" : transcript.text)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .padding(.top, 12)
        }
        .padding(18)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 20))
    }

    private var actions: some View {
        VStack(spacing: 10) {
            Button { model.togglePlayback(current) } label: {
                Label(
                    playback.playingID == current.id ? "Stop playback" : "Play recording",
                    systemImage: playback.playingID == current.id ? "stop.fill" : "play.fill"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            // Playback reconfigures the shared audio session and must not
            // compete with an active recording.
            .disabled(model.recorder.isRecording)

            if let audioURL = model.audioURLs[current.id] {
                ShareLink(item: audioURL) {
                    Label("Export recording", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }

            Button {
                Task { await model.retry(current, configuration: configuration) }
            } label: {
                Label("Transcribe again", systemImage: "arrow.clockwise")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(model.isWorking)

            Button(role: .destructive) { confirmsDelete = true } label: {
                Label("Delete recording and text", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
        }
    }

    private var durationText: String {
        guard let milliseconds = current.durationMilliseconds else { return "Unknown length" }
        let seconds = milliseconds / 1_000
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}
