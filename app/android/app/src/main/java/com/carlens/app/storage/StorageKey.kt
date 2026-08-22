package com.carlens.app.storage

import android.content.Context
import android.content.SharedPreferences
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.io.File
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Ключ шифрования локального хранилища (§9 спеки: «история в MMKV с шифрованием»).
 *
 * **Почему два уровня.** `react-native-mmkv` принимает ключ строкой, то есть ключ обязан
 * оказаться на JS-стороне. Отдать наружу ключ из `AndroidKeyStore` нельзя по построению:
 * `getEncoded()` у такого `SecretKey` возвращает `null` — в этом и смысл Keystore.
 * Поэтому ключ MMKV — обычные 32 случайных байта, а Keystore хранит несъёмную обёртку,
 * которой эти байты зашифрованы. Шифротекст лежит в `SharedPreferences`.
 *
 * Что схема даёт: файл истории бесполезен без ключа, ключ бесполезен без Keystore, а
 * Keystore не выносится с устройства. `android:allowBackup="false"` в манифесте не даёт
 * шифротексту уехать в облачный бэкап.
 *
 * Чего не даёт: на рутованном устройстве или под отладчиком процесс отдаёт и ключ, и
 * историю. Защищаются данные в покое — украденный телефон, вытащенный `/data`.
 */
object StorageKey {

    private const val АЛИАС_ОБЁРТКИ = "carlens_history_wrap"
    private const val ФАЙЛ_НАСТРОЕК = "carlens_storage"
    private const val КЛЮЧ_ШИФРОТЕКСТА = "wrapped_key"
    private const val КЛЮЧ_ВЕКТОРА = "wrap_iv"
    private const val ПРЕОБРАЗОВАНИЕ = "AES/GCM/NoPadding"
    private const val ДЛИНА_ТЕГА_БИТ = 128

    /** Длина ключа MMKV: 32 случайных байта, наружу уезжают Base64-строкой. */
    private const val ДЛИНА_КЛЮЧА_БАЙТ = 32

    /** Размер обёртки в Keystore — AES-256. С длиной ключа MMKV не связан. */
    private const val ДЛИНА_ОБЁРТКИ_БИТ = 256

    /** Итог инициализации: сам ключ и признак того, что старые данные пришлось бросить. */
    data class Результат(val ключBase64: String, val сброшен: Boolean)

    /**
     * Возвращает ключ MMKV, заводя его при первом запуске.
     *
     * Если шифротекст есть, но не расшифровывается — обёртка исчезла или инвалидирована
     * (сброс данных, восстановление на другом устройстве, смена блокировки экрана на
     * части прошивок), — старые данные нечитаемы навсегда. Тогда хранилище сносится
     * целиком, заводится новый ключ, а вызывающему сообщается `сброшен = true`. Открытым
     * текстом не пишется ничего ни при каком развитии событий.
     */
    fun получить(контекст: Context): Результат {
        val настройки = контекст.getSharedPreferences(ФАЙЛ_НАСТРОЕК, Context.MODE_PRIVATE)
        val шифротекст = настройки.getString(КЛЮЧ_ШИФРОТЕКСТА, null)
        val вектор = настройки.getString(КЛЮЧ_ВЕКТОРА, null)
        if (шифротекст != null && вектор != null) {
            val распакованный =
                runCatching { развернуть(шифротекст, вектор) }.getOrNull()
            if (распакованный != null) {
                return Результат(распакованный, сброшен = false)
            }
            // Ключ утрачен: данные уже не прочитать, и притворяться, что можно, нечем.
            снестиХранилище(контекст)
            настройки.edit().clear().apply()
            удалитьОбёртку()
            return Результат(завестиНовый(настройки), сброшен = true)
        }
        return Результат(завестиНовый(настройки), сброшен = false)
    }

    private fun завестиНовый(настройки: SharedPreferences): String {
        val сырой = ByteArray(ДЛИНА_КЛЮЧА_БАЙТ).also { SecureRandom().nextBytes(it) }
        val шифр = Cipher.getInstance(ПРЕОБРАЗОВАНИЕ)
        шифр.init(Cipher.ENCRYPT_MODE, обёртка())
        val шифротекст = шифр.doFinal(сырой)
        настройки
            .edit()
            .putString(КЛЮЧ_ШИФРОТЕКСТА, Base64.encodeToString(шифротекст, Base64.NO_WRAP))
            .putString(КЛЮЧ_ВЕКТОРА, Base64.encodeToString(шифр.iv, Base64.NO_WRAP))
            .apply()
        return Base64.encodeToString(сырой, Base64.NO_WRAP)
    }

    private fun развернуть(шифротекст: String, вектор: String): String {
        val хранилище = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val обёртка =
            хранилище.getKey(АЛИАС_ОБЁРТКИ, null) as? SecretKey
                ?: error("обёртка $АЛИАС_ОБЁРТКИ исчезла из AndroidKeyStore")
        val шифр = Cipher.getInstance(ПРЕОБРАЗОВАНИЕ)
        шифр.init(
            Cipher.DECRYPT_MODE,
            обёртка,
            GCMParameterSpec(ДЛИНА_ТЕГА_БИТ, Base64.decode(вектор, Base64.NO_WRAP)),
        )
        val сырой = шифр.doFinal(Base64.decode(шифротекст, Base64.NO_WRAP))
        return Base64.encodeToString(сырой, Base64.NO_WRAP)
    }

    private fun обёртка(): SecretKey {
        val хранилище = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (хранилище.getKey(АЛИАС_ОБЁРТКИ, null) as? SecretKey)?.let { return it }
        val генератор =
            KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        генератор.init(
            KeyGenParameterSpec.Builder(
                    АЛИАС_ОБЁРТКИ,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
                )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(ДЛИНА_ОБЁРТКИ_БИТ)
                // Аутентификация пользователя не требуется намеренно: приложение обязано
                // писать историю сразу после распознавания, а не после разблокировки.
                .setUserAuthenticationRequired(false)
                .build()
        )
        return генератор.generateKey()
    }

    private fun удалитьОбёртку() {
        runCatching {
            KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
                .deleteEntry(АЛИАС_ОБЁРТКИ)
        }
    }

    /**
     * Каталог MMKV по умолчанию — `filesDir/mmkv`. Сносится целиком: файлы, зашифрованные
     * утраченным ключом, не откроются никогда, и оставлять их — только занимать место и
     * путать следующий запуск.
     */
    private fun снестиХранилище(контекст: Context) {
        runCatching { File(контекст.filesDir, "mmkv").deleteRecursively() }
    }
}
