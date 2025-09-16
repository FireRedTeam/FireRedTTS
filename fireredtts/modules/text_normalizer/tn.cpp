
#include <pybind11/pybind11.h>
#include "include/tn.h"
#include "include/common/rede_error.h"
#include "include/common/rede_logging.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <vector>

namespace py = pybind11;
static Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "tts_env");

struct RedTN {
    RedTN(const std::string &datapath) : datapath(datapath) {
        rede::REDE_ERROR err;
        err = rede::tn::tn_init(datapath, env);
        if (REDE_FAILED(err))
        {
            std::cerr << "RedTN::init error" << std::endl;
        }
    }

    std::string tn(const std::string &text) {
        rede::REDE_ERROR err;
        std::string output = "";
        err = rede::tn::tn_text(text, output, true, true);
        if (REDE_FAILED(err))
        {
            std::cerr << "RedTN::process error" << std::endl;
	    return "";
        }
        rede::tn::StringReplace(output, std::string("壹"), std::string("一"));
        return output;
    }

    std::string datapath;
};

PYBIND11_MODULE(redtn, m) {
    m.doc() = "redtts tn plugin"; // optional module docstring

    py::class_<RedTN>(m, "RedTN")
        .def(py::init<const std::string &>())
        .def("tn", &RedTN::tn)
        .def("__repr__",
            [](const RedTN &a) {
                return "<redtn.RedTN initialized from '" + a.datapath + "'>";
            }
        );

}
