/*!
 *  \file       rede_error.h
 *  \brief      Error type definition
 *  \author     luanbu@xiaohongshu.com
 *  \copyright  xiaohongshu
 *  \version    0.0.0
 *  \date       2022/02/28
 */

#ifndef _REDE_COMMON_ERROR_H_
#define _REDE_COMMON_ERROR_H_

namespace rede {

enum class REDE_ERROR : int
{
    // API ERROR CODES
    REDE_OK                             = 0,    ///< OK.
    REDE_E_FAILED                       = 1,    ///< Failure.
    REDE_E_NOTINITIALIZED               = 2,    ///< API not appropriately initialized.
    REDE_E_FILEREAD                     = 3,    ///< Error reading file.
    REDE_E_FILEWRITE                    = 4,    ///< Error writing file.
    REDE_E_FILECLOSE                    = 5,    ///< Error closing file.
    REDE_E_INVALIDARG                   = 6,    ///< Invalid argument.
    REDE_E_BUFFERTOOSMALL               = 7,    ///< Buffer is too small.
    REDE_E_OUTOFMEMORY                  = 8,    ///< Out of memory.
    REDE_E_MODULENOTFOUND               = 9,    ///< Failed to locate a module.
    REDE_E_INVALIDPARAM                 = 10,   ///< Parameter value invalid.
    REDE_E_OBJNOTFOUND                  = 11,   ///< Object not found.
    REDE_E_VERSION                      = 12,   ///< Version mismatch.

    // SPECIFIC ERROR CODES
    REDE_E_CONVERSIONFAILED             = 20,   ///< Encoding conversion failed.
    REDE_E_INVALIDCHAR                  = 21,   ///< Invalid UTF-8 character.
    REDE_E_NOINPUTTEXT                  = 22,   ///< No input text.
    REDE_E_LANGUAGENOTFOUND             = 23,   ///< Specified language not found.
    REDE_E_VOICENOTFOUND                = 24,   ///< Specified voice not found.
    REDE_E_USERSTOP                     = 25,   ///< Processing stopped at user's request.
    REDE_E_ILLEGALDICTFORMAT            = 26,   ///< Illegal text dictionary format.
    REDE_E_EMPTYOUTPUT                  = 27,   ///< No valid output.

    // WARNING CODES
    REDE_W_FAILED                       = 101,   ///< [Warn] Failed.
    REDE_W_NOINPUTTEXT                  = 102,   ///< [Warn] No input text.
    REDE_W_EOF                          = 103,   ///< [Warn] End of File reached.
    REDE_W_CHARSKIPPED                  = 104   ///< [Warn] A character was skipped during encoding conversion.
};

#define REDE_FAILED(err_code) ((int)err_code > 0 && (int)err_code < 100)

} // namespace rede

#endif
