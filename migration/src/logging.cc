#include <array>
#include <cstring>
#include <iostream>
#include <vector>

#include <openssl/aes.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "migration/logging.h"

static inline std::string base64_encode(const unsigned char* data, size_t len) {
  static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  out.reserve(((len / 3) + 1) * 4);

  unsigned int val = 0;
  int valb         = -6;

  for (size_t i = 0; i < len; i++) {
    val = (val << 8) + data[i];
    valb += 8;
    while (valb >= 0) {
      out.push_back(b64[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(b64[((val << 8) >> (valb + 8)) & 0x3F]);

  while (out.size() % 4) out.push_back('=');

  return out;
}

template <typename Mutex>
class rsa_encrypt_base64_sink : public spdlog::sinks::base_sink<Mutex> {
 public:
  rsa_encrypt_base64_sink() {
    // Load RSA pubkey from string
    const char* public_key_str = R"(-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqzudEYdMdilhTiONl10Z
+uliDOczfjspRZFgSt4ZCJe6+AhnE/GBLXcaB1Rk/sQtKmOPoxIrbZIVhRqtT1um
02orUTGUdbdO9mxALWA72e+wzI5y7VPRxPc6bgeJ57voMyATFxyH+DTjJ8pXTLKf
9I3UNgyG6pXO6N/jJdIQsi8zR6pM7RVAZ+UHMho23FI3A27imqdRYMH4P/aAmnsK
yMdWpe8l0Idb5SpKhADqgoR+fKqOOg/jDRLKXv8z8VbLuRbi7uC/CuZi3GPTOodo
yHgzaKPwrDm08mSnvkids3Dlq+pAoCh0abbG/RGUg+j5QTm+apvtNaN8nxYnunDg
lwIDAQAB
-----END PUBLIC KEY-----)";

    /*
    -----BEGIN PRIVATE KEY-----
    MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCrO50Rh0x2KWFO
    I42XXRn66WIM5zN+OylFkWBK3hkIl7r4CGcT8YEtdxoHVGT+xC0qY4+jEittkhWF
    Gq1PW6bTaitRMZR1t072bEAtYDvZ77DMjnLtU9HE9zpuB4nnu+gzIBMXHIf4NOMn
    yldMsp/0jdQ2DIbqlc7o3+Ml0hCyLzNHqkztFUBn5QcyGjbcUjcDbuKap1Fgwfg/
    9oCaewrIx1al7yXQh1vlKkqEAOqChH58qo46D+MNEspe/zPxVsu5FuLu4L8K5mLc
    Y9M6h2jIeDNoo/CsObTyZKe+SJ2zcOWr6kCgKHRptsb9EZSD6PlBOb5qm+01o3yf
    Fie6cOCXAgMBAAECggEAG8OdvdLZ+KrQUT193tYaDqS64OwNXba1WXIEp/8HwMHs
    Gf2FdnwJKFhlywcfHRlYAsXZQLqtz1AJgecyb2n6odBW17fEolWnowMoYMh4IT02
    cI8vCVY27stGGRh11DI+qH4sj5RxDjNycD2PFtY7uLPVKIeyyeSig6s8XHc0J9Uv
    9fU5v32S3x5ShCkBc6TKv9tX7BtDcY/DveTMIzNlBhV7ocs7Gf1nlPvtMIR7SSJu
    PGDxv+7X8/dcRCEBPyC2d5B7CIQSnXmM4H0xklZRxykJYBeyBuMB2iC0m+86QPgA
    ZuH0yumT6n/zkZpLrta8sQ9Nl/LAkrxtWACltzqUsQKBgQDi6Px/SelXTiWNEWk0
    JZQfx9L64XPQnw5U2qhg6sbDDuYWuSF81xROUpZgM8olJwpjvglkGmf91n2NOnmq
    hSHsT62Sl70dwCgjCboMHaHRBaxPGX2rKDqXdL9ID4XZmYgv94ot8okPTVJjNhvm
    2Ib7Vmg+09V+l/KSkxFSk8fyyQKBgQDBL1fs9GIEuXOApx1+fopJEdlPh3bz6pI9
    nnwEPRUAIcO4it5LIFH8e4zuxoqQgr6Hwu3yowSwYMRW4FAumqWpMS0OPhcE62cW
    WFnbrpH+cx+m1WhPdZCuDNiZBzQBdNzhUf6j7TTHqmudjOMpWZNLJnGl9cQZSBZe
    MiGWjquIXwKBgFO50GFPbnAuf6CbygvZydwoKWs1ATz7U5hvzi1ks86JktDTos2j
    tvRneOEqeu5Wh3jiSCjNrY12NYGFEBuhYDEH/W3X24o8uxKipimOTYUI6NmO+FXN
    VEFKbMI0KBlwk1XPqwblNTmWOE4vSwBU6QmYioKUO3SosHLxHTUxHlgxAoGATdKl
    qAY26lJPDlfEEO4nBRKUqW4X5GDtsrcCnK6CpD/12YTP0hHeFUksWBBRR6/z0zsa
    ojE7tVX2Ik1Q38Va0RLHZMJsgYXXTHAhGtdzZr631HyJ/eCNfSAdrV/yele6l2Zx
    n1XyejDUE27rIAA+zvpYtBOSgODCagXl9AHbZh8CgYEApHhOLZFpAEwQrQq8mRI3
    ISB2CKexgNqF7Pj1/FCwKgkFioUvf5CCcepjUdsK2q2QFrOLGoUuqZnulYaORpxp
    +zZX7AyVAyrw901k27NdCMVU0KkIkIUN6fJYE3qCazRMesjRR/gF9qv6elc00KWY
    544r3mOSwEAWGG90BO4Gv54=
    -----END PRIVATE KEY-----
    */

    BIO* bio = BIO_new_mem_buf(public_key_str, -1);
    if (!bio) throw std::runtime_error("Failed to create BIO");

    pkey_ = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);

    if (!pkey_) throw std::runtime_error("Failed to load RSA public key");

    aes_key_.resize(kAesKeySize);
    if (RAND_bytes(aes_key_.data(), static_cast<int>(aes_key_.size())) != 1) {
      throw std::runtime_error("Failed to generate AES key");
    }

    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(pkey_, nullptr);
    if (!ctx) throw std::runtime_error("Failed to create EVP_PKEY_CTX");

    if (EVP_PKEY_encrypt_init(ctx) <= 0) {
      EVP_PKEY_CTX_free(ctx);
      throw std::runtime_error("Failed to init public key encrypt");
    }
    if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING) <= 0) {
      EVP_PKEY_CTX_free(ctx);
      throw std::runtime_error("Failed to set RSA padding");
    }

    size_t enc_len = 0;
    if (EVP_PKEY_encrypt(ctx, nullptr, &enc_len, aes_key_.data(), aes_key_.size()) <= 0) {
      EVP_PKEY_CTX_free(ctx);
      throw std::runtime_error("Failed to determine encrypted key length");
    }

    std::vector<unsigned char> encrypted_key(enc_len);
    if (EVP_PKEY_encrypt(ctx, encrypted_key.data(), &enc_len, aes_key_.data(), aes_key_.size()) <= 0) {
      EVP_PKEY_CTX_free(ctx);
      throw std::runtime_error("Failed to encrypt AES key");
    }
    encrypted_key.resize(enc_len);
    EVP_PKEY_CTX_free(ctx);

    header_text_ = base64_encode(encrypted_key.data(), encrypted_key.size());
  }

  ~rsa_encrypt_base64_sink() {
    if (pkey_) EVP_PKEY_free(pkey_);
  }

 protected:
  std::vector<unsigned char> aes_encrypt(const std::string& plain) {
    std::vector<unsigned char> iv(kAesIvSize);
    if (RAND_bytes(iv.data(), static_cast<int>(iv.size())) != 1) {
      throw std::runtime_error("Failed to generate IV");
    }

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) throw std::runtime_error("Failed to create cipher ctx");

    std::vector<unsigned char> ciphertext(plain.size() + kAesBlockSize);
    int len = 0, total_len = 0;

    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, aes_key_.data(), iv.data()) != 1) {
      EVP_CIPHER_CTX_free(ctx);
      throw std::runtime_error("EncryptInit failed");
    }

    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, reinterpret_cast<const unsigned char*>(plain.data()), static_cast<int>(plain.size())) != 1) {
      EVP_CIPHER_CTX_free(ctx);
      throw std::runtime_error("EncryptUpdate failed");
    }
    total_len = len;

    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + total_len, &len) != 1) {
      EVP_CIPHER_CTX_free(ctx);
      throw std::runtime_error("EncryptFinal failed");
    }
    total_len += len;
    ciphertext.resize(static_cast<size_t>(total_len));

    EVP_CIPHER_CTX_free(ctx);

    // 返回 iv + ciphertext
    std::vector<unsigned char> out;
    out.reserve(iv.size() + ciphertext.size());
    out.insert(out.end(), iv.begin(), iv.end());
    out.insert(out.end(), ciphertext.begin(), ciphertext.end());
    return out;
  }

  void sink_it_(const spdlog::details::log_msg& msg) override {
    // format log text
    spdlog::memory_buf_t formatted;
    this->formatter_->format(msg, formatted);

    std::string plain(formatted.data(), formatted.size());

    if (!header_written_) {
      std::cout << header_text_ << std::endl;
      header_written_ = true;
    }

    std::vector<unsigned char> encrypted = aes_encrypt(plain);
    std::string encoded                  = base64_encode(encrypted.data(), encrypted.size());

    // print to console
    std::cout << encoded << std::endl;
  }

  void flush_() override { std::cout.flush(); }

 private:
  static constexpr size_t kAesKeySize   = 32;  // AES-256
  static constexpr size_t kAesIvSize    = 16;
  static constexpr size_t kAesBlockSize = 16;

  EVP_PKEY* pkey_ = nullptr;
  std::vector<unsigned char> aes_key_;
  std::string header_text_;
  bool header_written_ = false;
};

void InitSpdLog() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_sink    = std::make_shared<rsa_encrypt_base64_sink<std::mutex>>();

  std::vector<spdlog::sink_ptr> sinks;
#if !defined(NDEBUG) || defined(MIGRATION_PLAINTEXT_LOGGING)
  sinks.push_back(console_sink);
#else
  sinks.push_back(file_sink);
#endif

  auto logger = std::make_shared<spdlog::logger>("", sinks.begin(), sinks.end());
  spdlog::set_default_logger(logger);
}
