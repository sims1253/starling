# Compatibility entry point for existing native builds and releases.
cmake_minimum_required(VERSION 3.18)
project(starling LANGUAGES C CXX)
add_subdirectory(backends/native)
