package com.carlens.vision

import android.content.res.AssetManager

class AssetMismatchException(message: String) : IllegalStateException(message)

object Assets {

    /**
     * Прочитать ассет целиком и сверить sha256.
     *
     * Сверка ловит рассинхронизацию модели и индекса — набор от разных прогонов рядом
     * друг с другом, — а не подмену APK: эталон лежит в том же артефакте, что и
     * проверяемый файл (§10 дизайна). Рассинхронизация не падает сама: приложение
     * уверенно показывает неправильные подписи.
     */
    fun readVerified(assets: AssetManager, name: String, expectedSha256: String?): ByteArray =
        verified(assets.open(name).use { it.readBytes() }, name, expectedSha256)

    /**
     * Сама сверка, отдельно от чтения: `AssetManager` на JVM недоступен, а ошибиться
     * можно ровно здесь, поэтому проверяемая часть обязана быть без Android.
     */
    fun verified(data: ByteArray, name: String, expectedSha256: String?): ByteArray {
        if (expectedSha256 != null) {
            val фактический = Embeddings.sha256(data)
            if (фактический != expectedSha256) {
                throw AssetMismatchException(
                    "sha256 ассета $name: $фактический против ожидаемого $expectedSha256 — " +
                        "модели разложены от разных прогонов"
                )
            }
        }
        return data
    }
}
