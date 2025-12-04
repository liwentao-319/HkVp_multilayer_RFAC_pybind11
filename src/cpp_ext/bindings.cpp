#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hk_stacking.h"
#include "mathfuns.h"

namespace py = pybind11;

PYBIND11_MODULE(hk_stacking, m) {
    m.doc() = "Hk stacking for multilayer Vp model";

    py::class_<Hk_stacking_multilayer_Vp>(m, "Hk_stacking_multilayer_Vp")
        .def(py::init<
            const size_t& ,
            const float& ,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<int>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<int>&,
            py::array_t<float>&
        >()) 
        .def("Hk_stacking", &Hk_stacking_multilayer_Vp::Hk_stacking)
        .def("Hk_stacking_PWS", &Hk_stacking_multilayer_Vp::Hk_stacking_PWS)
        .def("Hk_stacking_PWS1", &Hk_stacking_multilayer_Vp::Hk_stacking_PWS1)
        .def("Hk_stacking_PWS2", &Hk_stacking_multilayer_Vp::Hk_stacking_PWS2)
        .def("test_constructor", &Hk_stacking_multilayer_Vp::test_constructor)
        .def("get_stacked_image", &Hk_stacking_multilayer_Vp::get_stacked_image)
        .def("get_phasetimes", &Hk_stacking_multilayer_Vp::get_phasetimes)
        .def("get_Hs_optimal", &Hk_stacking_multilayer_Vp::get_Hs_optimal)
        .def("get_Ks_optimal", &Hk_stacking_multilayer_Vp::get_Ks_optimal)
        .def("get_Hs", &Hk_stacking_multilayer_Vp::get_Hs)
        .def("get_Ks", &Hk_stacking_multilayer_Vp::get_Ks)
        .def("get_Vps", &Hk_stacking_multilayer_Vp::get_Vps)
        .def("get_weights", &Hk_stacking_multilayer_Vp::get_weights)
        .def("print_weights",&Hk_stacking_multilayer_Vp::print_weights);

    m.def("cal_hilbert_phase", &cal_hilbert_phase, "Cal Hilbert Phase", py::arg("input"));
    
    // py::register_exception<std::invalid_argument>(m, "InvalidArgumentError");
}