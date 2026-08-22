package com.carlens.app

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import java.security.MessageDigest

/**
 * Сверка отпечатка сертификата, которым подписан установленный APK, с тем, что вшит в
 * `BuildConfig` на сборке.
 *
 * **Чем это не является.** Защитой от подмены моделей — нет. Перепаковщик распаковывает
 * APK, меняет модель, меняет там же ожидаемый отпечаток, подписывает своим ключом и
 * ставит: проверка сравнит его подпись с его же подставленным эталоном и сойдётся.
 * Приложение не может быть корнем доверия для самого себя — всё, чем оно проверяет,
 * лежит внутри того же артефакта. От изменения файлов уже установленного APK защищает
 * сама подпись Android на уровне системы; для защиты от перепакованного APK нужен внешний
 * якорь доверия (сервер аттестации, Play Integrity), а его в MVP нет и не планируется:
 * §9 спеки — всё on-device, сети нет вовсе.
 *
 * **Чем является.** Контролем сборочной конфигурации: ловит рассинхрон между тем, чем
 * сборка подписывалась, и тем, что стоит на устройстве.
 */
object SigningIdentity {

    private const val ТЕГ = "CarlensSigning"

    /**
     * Пустой ожидаемый отпечаток означает «на машине сборки не было keystore» — проверка
     * тогда молча выключена. Расхождение непустых значений роняет запуск: собранное и
     * установленное разошлись, и дальше идти незачем.
     */
    fun проверить(контекст: Context, ожидаемый: String = BuildConfig.SIGNING_CERT_SHA256) {
        if (ожидаемый.isEmpty()) {
            Log.i(ТЕГ, "отпечаток подписи не вшит на сборке — проверка выключена")
            return
        }
        val фактический = отпечатокУстановленного(контекст)
        if (фактический == null) {
            Log.w(ТЕГ, "подпись установленного пакета не читается — проверка пропущена")
            return
        }
        check(фактический.equals(ожидаемый, ignoreCase = true)) {
            "Подпись установленного пакета не совпадает со сборочной: " +
                "ожидался $ожидаемый, установлен $фактический"
        }
    }

    private fun отпечатокУстановленного(контекст: Context): String? {
        val менеджер = контекст.packageManager
        val имя = контекст.packageName
        val сертификаты =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val пакет =
                    менеджер.getPackageInfo(имя, PackageManager.GET_SIGNING_CERTIFICATES)
                // `signingInfo` даёт либо один текущий сертификат, либо цепочку ротации.
                // Берётся первый текущий: ротацией ключа этот проект не пользуется.
                пакет.signingInfo?.apkContentsSigners
            } else {
                @Suppress("DEPRECATION")
                менеджер.getPackageInfo(имя, PackageManager.GET_SIGNATURES).signatures
            }
        val первый = сертификаты?.firstOrNull() ?: return null
        return MessageDigest.getInstance("SHA-256")
            .digest(первый.toByteArray())
            .joinToString("") { "%02x".format(it) }
    }
}
