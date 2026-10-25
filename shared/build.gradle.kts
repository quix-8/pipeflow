plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.kotlinSerialization)
}

kotlin {
    jvm()


    sourceSets {
        commonMain.dependencies {
            implementation(libs.compose.runtime)
            implementation(libs.compose.foundation)
            implementation(libs.compose.material3)
            implementation(libs.compose.ui)
            implementation(libs.compose.components.resources)
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.lifecycle.viewmodelCompose)
            implementation(libs.androidx.lifecycle.runtimeCompose)
            implementation(libs.kotlinx.serialization.json)
            implementation("io.github.vinceglb:filekit-dialogs:0.16.0")
            implementation("io.github.vinceglb:filekit-dialogs-compose:0.16.0")
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
        jvmMain.dependencies {
            implementation(compose.desktop.currentOs)
            // ... тут обычно лежит kotlinx-coroutines-swing

            // Дописываем наши JVM-библиотеки для OCR и картинок:
            implementation(libs.tess4j)
            implementation(libs.rapidocr.onnx)
            implementation(libs.scrimage.core)
            implementation(libs.scrimage.filters)
            implementation(libs.slf4j.simple)
        }
    }
}