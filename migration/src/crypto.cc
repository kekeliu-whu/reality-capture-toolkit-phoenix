#include <glog/logging.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

#include <string>
#include <iostream>

#include "migration/crypto.h"

namespace {

	const unsigned char KEY[33] = "12345678901234567890123456789012";  // 32 bytes AES-256
	const unsigned char IV[17] = "1234567890123456";  // 16 bytes IV

}

// Base64 encode
std::string Base64Encode(const std::string& input) {
	BIO* bio, * b64;
	BUF_MEM* bufferPtr;
	b64 = BIO_new(BIO_f_base64());
	bio = BIO_new(BIO_s_mem());
	BIO_push(b64, bio);
	BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);  // no new line
	BIO_write(b64, input.data(), (int)input.size());
	BIO_flush(b64);
	BIO_get_mem_ptr(b64, &bufferPtr);
	std::string encoded(bufferPtr->data, bufferPtr->length);
	BIO_free_all(b64);
	return encoded;
}

// Base64 decode
std::string Base64Decode(const std::string& input) {
	BIO* bio, * b64;
	char* buffer = (char*)malloc(input.size());
	memset(buffer, 0, (int)input.size());
	b64 = BIO_new(BIO_f_base64());
	bio = BIO_new_mem_buf(input.data(), (int)input.size());
	BIO_push(b64, bio);
	BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
	int len = BIO_read(b64, buffer, (int)input.size());
	std::string decoded(buffer, len);
	free(buffer);
	BIO_free_all(b64);
	return decoded;
}

// AES encrypt
std::string EncryptAES(const std::string& plaintext) {
	EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
	std::string ciphertext;
	ciphertext.resize(plaintext.size() + 16);  // padding

	int len;
	EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, KEY, IV);
	EVP_EncryptUpdate(ctx, (unsigned char*)&ciphertext[0], &len, (const unsigned char*)plaintext.data(), (int)plaintext.size());

	int totalLen = len;
	EVP_EncryptFinal_ex(ctx, (unsigned char*)&ciphertext[0] + len, &len);
	totalLen += len;
	EVP_CIPHER_CTX_free(ctx);

	ciphertext.resize(totalLen);
	return Base64Encode(ciphertext);
}

// AES decrypt
std::string DecryptAES(const std::string& base64_cipher) {
	std::string encrypted = Base64Decode(base64_cipher);
	EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
	std::string plaintext;
	plaintext.resize(encrypted.size());

	int len;
	EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, KEY, IV);
	EVP_DecryptUpdate(ctx, (unsigned char*)&plaintext[0], &len, (const unsigned char*)encrypted.data(), (int)encrypted.size());

	int totalLen = len;
	EVP_DecryptFinal_ex(ctx, (unsigned char*)&plaintext[0] + len, &len);
	totalLen += len;
	EVP_CIPHER_CTX_free(ctx);

	plaintext.resize(totalLen);
	return plaintext;
}

void EncryptedLogSink::send(google::LogSeverity severity, const char* full_filename, const char* base_filename,
	int line, const struct ::tm* tm_time, const char* message, size_t message_len) {
	std::string msg(message, message_len);
	std::string encrypted = EncryptAES(msg);
	std::cout << "[EncryptedLog] " << encrypted << std::endl;
}
