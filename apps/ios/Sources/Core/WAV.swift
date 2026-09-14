import Foundation

public enum WAVValidationError: LocalizedError, Equatable {
    case malformed
    case unsupportedFormat

    public var errorDescription: String? {
        switch self {
        case .malformed: "The recording is not a complete WAV file."
        case .unsupportedFormat: "The recording must be mono 16 kHz, 16-bit PCM WAV."
        }
    }
}

/// Validate the exact audio subset shared by Starling's native and standard
/// HTTP routes. Chunk sizes are checked against actual bytes before reading.
public func validateStarlingWAV(_ data: Data) throws {
    guard data.count >= 44,
          data.ascii(at: 0, count: 4) == "RIFF",
          data.ascii(at: 8, count: 4) == "WAVE"
    else { throw WAVValidationError.malformed }
    let declaredFileSize = Int(data.littleEndianUInt32(at: 4)) + 8
    guard declaredFileSize <= data.count else { throw WAVValidationError.malformed }

    var offset = 12
    var validFormat = false
    var validData = false
    while offset <= data.count - 8 {
        let id = data.ascii(at: offset, count: 4)
        let size = Int(data.littleEndianUInt32(at: offset + 4))
        let body = offset + 8
        guard body <= data.count, size <= data.count - body else {
            throw WAVValidationError.malformed
        }
        if id == "fmt " {
            guard size >= 16 else { throw WAVValidationError.malformed }
            let encoding = data.littleEndianUInt16(at: body)
            let channels = data.littleEndianUInt16(at: body + 2)
            let sampleRate = data.littleEndianUInt32(at: body + 4)
            let bits = data.littleEndianUInt16(at: body + 14)
            guard encoding == 1, channels == 1, sampleRate == 16_000, bits == 16 else {
                throw WAVValidationError.unsupportedFormat
            }
            validFormat = true
        } else if id == "data" {
            guard size > 0, size.isMultiple(of: 2) else { throw WAVValidationError.malformed }
            validData = true
        }
        offset = body + size + (size.isMultiple(of: 2) ? 0 : 1)
    }
    guard validFormat, validData else { throw WAVValidationError.malformed }
}

private extension Data {
    func ascii(at offset: Int, count: Int) -> String {
        guard offset >= 0, count >= 0, offset + count <= self.count else { return "" }
        return String(decoding: self[offset ..< offset + count], as: UTF8.self)
    }

    func littleEndianUInt16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) | (UInt16(self[offset + 1]) << 8)
    }

    func littleEndianUInt32(at offset: Int) -> UInt32 {
        UInt32(self[offset])
            | (UInt32(self[offset + 1]) << 8)
            | (UInt32(self[offset + 2]) << 16)
            | (UInt32(self[offset + 3]) << 24)
    }
}
