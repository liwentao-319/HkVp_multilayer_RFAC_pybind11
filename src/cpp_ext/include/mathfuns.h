#pragma once
#include <cmath>
#include <stdexcept>
#include <complex>
#include <fftw3.h>
#include <vector>
#include <pybind11/numpy.h> 
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

inline float sqrtf32(float value) {
    if (value < 0) {
        throw std::invalid_argument("Cannot compute square root of a negative number.");
    }
    return std::sqrt(value);
}


inline float squaref32(float value) {
    return value * value;
}  

inline int sgn(float value) {
    return (value > 0) - (value < 0);
}



/**
 * Computes the Hilbert transform of a real-valued signal
 * @param in Pointer to input float array (real signal)
 * @param out Pointer to output complex<float> array (analytic signal)
 * @param n Length of input/output arrays
 */

void hilbert_transform(const float* in, std::complex<float>* out, size_t n);

py::array_t<float> cal_hilbert_phase( py::array_t<float> input);