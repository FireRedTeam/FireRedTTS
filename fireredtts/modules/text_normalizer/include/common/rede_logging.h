/*!
 *  \file       rede_logging.h
 *  \brief      Defines logging functions and ENV variables
 *  \author     luanbu@xiaohongshu.com
 *  \copyright  xiaohongshu
 *  \version    0.0.0
 *  \date       2022/02/28
 */

#pragma once

#ifndef _REDE_TTS_COMMON_LOGGING_H_
#define _REDE_TTS_COMMON_LOGGING_H_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string.h>
#include <iomanip>

#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)

namespace rede {
namespace logging {

enum LogLevel
{
    FATAL = 0,
    ERROR = 1,
    WARN = 2,
    PROF = 3,
    INFO = 4,
    DEBUG = 5,
    TRACE = 6,
    LOGLEVEL_MAX = 6
};

static std::string LogLevel2String(LogLevel lvl)
{
    switch (lvl){
        case FATAL:
            return "0";
        case ERROR:
            return "1";
        case WARN:
            return "2";
        case PROF:
            return "3";
        case INFO:
            return "4";
        case DEBUG:
            return "5";
        case TRACE:
        default:
            return "6";
    }
}

//static const char *log_level_str = getenv("REDE_LOG_LEVEL");
//extern int log_level = (log_level_str && atoi(log_level_str) >= 0 && atoi(log_level_str) <= LOGLEVEL_MAX) ? atoi(log_level_str) : PROF;
static int log_level = 1;

class LogWrapper {
public:
    explicit LogWrapper(LogLevel level)
        : level_(level) { need_log_ = (level <= log_level); }
    std::stringstream& getOutStreamRef() { return outStream_; }
    ~LogWrapper() {
        if (need_log_) {
            switch (level_) {
                case TRACE:
                case DEBUG:
                case INFO:
                case WARN:
                case PROF:
                    std::cout << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                    break;
                case ERROR:
                    std::cerr << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                    break;
                case FATAL:
                    std::cerr << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                    std::flush(std::cerr);
                    std::abort();
                    break;
                default:
                    std::cout << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
            }
        }
    }
private:
    std::stringstream outStream_;
    LogLevel level_;
    bool need_log_;
};

class LogStringWrapper
{
public:
    explicit LogStringWrapper(LogLevel level, std::string &log)
        : log_(log), level_(level) { need_log_ = (level <= log_level); }
    std::stringstream &getOutStreamRef() { return outStream_; }
    ~LogStringWrapper()
    {
        if (need_log_)
        {
            switch (level_)
            {
            case TRACE:
            case DEBUG:
            case INFO:
                log_ = log_ + "[" + LogLevel2String(level_) + "]" + outStream_.str() + "\n";
                break;
            case WARN:
            case PROF:
                log_ = log_ + "[" + LogLevel2String(level_) + "]" + outStream_.str() + "\n";
                std::cout << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                break;
            case ERROR:
                log_ = log_ + "[" + LogLevel2String(level_) + "]" + outStream_.str() + "\n";
                std::cerr << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                break;
            case FATAL:
                log_ = log_ + "[" + LogLevel2String(level_) + "]" + outStream_.str() + "\n";
                std::cerr << "[" << LogLevel2String(level_) << "]" << outStream_.str() << std::endl;
                std::flush(std::cerr);
                std::abort();
                break;
            default:
                log_ = log_ + "[" + LogLevel2String(level_) + "]" + outStream_.str() + "\n";
            }
        }
    }

private:
    std::stringstream outStream_;
    std::string &log_;
    LogLevel level_;
    bool need_log_;
};

} // namespace logging

#define REDE_LOG(level) logging::LogWrapper(logging::level).getOutStreamRef() \
                        << "[" << std::setw(30) << __FILENAME__ << "." << std::setw(5) << __LINE__ << "." << std::setw(30) << __FUNCTION__ << "] "

#define REDE_STRING_LOG(level, log) logging::LogStringWrapper(logging::level, log).getOutStreamRef() \
                        << "[" << std::setw(30) << __FILENAME__ << "." << std::setw(5) << __LINE__ << "." << std::setw(30) << __FUNCTION__ << "] "

} // namespace rede

#endif // _REDE_TTS_COMMON_LOGGING_H_
