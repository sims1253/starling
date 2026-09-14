import SwiftUI

/// Shared app colors. The palette mirrors the desktop client while keeping
/// SwiftUI's native controls and automatic contrast behavior.
enum StarlingTheme {
    static let charcoal = Color(red: 0x17 / 255.0, green: 0x18 / 255.0, blue: 0x13 / 255.0)
    static let panel = Color(red: 0x20 / 255.0, green: 0x21 / 255.0, blue: 0x1C / 255.0)
    static let paper = Color(red: 0xE8 / 255.0, green: 0xE4 / 255.0, blue: 0xD9 / 255.0)
    static let ink = Color(red: 0x24 / 255.0, green: 0x25 / 255.0, blue: 0x1E / 255.0)
    static let muted = Color(red: 0x98 / 255.0, green: 0x99 / 255.0, blue: 0x8E / 255.0)
    static let lime = Color(red: 0xD3 / 255.0, green: 0xE8 / 255.0, blue: 0x89 / 255.0)
    static let coral = Color(red: 0xFF / 255.0, green: 0x74 / 255.0, blue: 0x5B / 255.0)
    static let amber = Color(red: 0xEF / 255.0, green: 0xC2 / 255.0, blue: 0x6B / 255.0)
}
