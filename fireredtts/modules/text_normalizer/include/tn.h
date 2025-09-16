
#ifndef _REDE_TTS_TN_TN_H_
#define _REDE_TTS_TN_TN_H_

#include "common/rede_error.h"
#include <string>
#include <stdint.h>

#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#if defined(__GNUC__)
#define FE_API __attribute__((visibility ("default")))
#else
#define FE_API
#endif

namespace rede::tn {
	enum FeComp
	{
		COMP_PUNCT = 0,
		COMP_TRUNC = 1,
		COMP_RETTT = 2,
		COMP_TN_BEFOREWS = 3,
		COMP_ALL = 100
	};

	FE_API void StringReplace(std::string &strBase, const std::string &strSrc, const std::string &strDes);

	// FE_API REDE_ERROR tn_init(const std::string& data_dir);
	FE_API REDE_ERROR tn_init(const std::string &data_dir, Ort::Env &env);

	//get symbols cleaned after rettt
	FE_API REDE_ERROR clean_text(const std::string& input, std::string& result, bool addNewLine=false);
	
	//get text normailized output
	FE_API REDE_ERROR tn_text(const std::string& input, std::string& result, bool addNewLine=false, bool addLang=false);
 
} // ~ namespace rede
#endif /* _REDE_TTS_TN_TN_H_ */
