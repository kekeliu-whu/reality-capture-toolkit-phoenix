
#pragma once
#include <spdlog/sinks/basic_file_sink.h>

#include <spdlog/common.h>
#include <spdlog/details/os.h>

namespace spdlog {
    namespace sinks {
        template<typename Mutex>
        class cryptographic_sink final : public base_sink<Mutex> {
        public:
            explicit cryptographic_sink(const filename_t &filename, bool truncate = false,
                                        const file_event_handlers &event_handlers = {}) : file_helper_{event_handlers} {
                file_helper_.open(filename, truncate);
            }

            const filename_t &filename() const {
                return file_helper_.filename();
            }

        protected:
            void sink_it_(const details::log_msg &msg) override {
                memory_buf_t formatted;
                base_sink<Mutex>::formatter_->format(msg, formatted);
                constexpr auto val = (1 << 6);
                for (char &c: formatted) {
                    c ^= val;
                }
                file_helper_.write(formatted);
            }

            void flush_() override {
                file_helper_.flush();
            }

        private:
            details::file_helper file_helper_;
        };

        using cryptographic_sink_mt = cryptographic_sink<std::mutex>;
        using cryptographic_sink_st = cryptographic_sink<details::null_mutex>;
    }
}

