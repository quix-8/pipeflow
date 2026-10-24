package com.quix.pipeflow

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform