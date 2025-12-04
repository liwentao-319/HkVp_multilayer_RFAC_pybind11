#include "mathfuns.h"

void hilbert_transform(const float* in, std::complex<float>* out, size_t n) {
    // Allocate FFTW arrays (using single precision floats)
    fftwf_complex* fft_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_complex* fft_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n);
    
    // Prepare input data (real signal in real part, 0 in imaginary part)
    for (size_t i = 0; i < n; ++i) {
        fft_in[i][0] = in[i];  // Real part
        fft_in[i][1] = 0.0f;   // Imaginary part
    }

    // Create and execute FFT plan
    fftwf_plan plan = fftwf_plan_dft_1d(n, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Frequency domain processing for Hilbert transform:
    // - Zero out DC and Nyquist components
    // - Double positive frequencies
    // - Zero negative frequencies
    for (size_t i = 0; i < n; ++i) {
        if (i == 0 || i == n / 2) {
            fft_out[i][0] = 0.0f;
            fft_out[i][1] = 0.0f;
        } else if (i < n / 2) {
            fft_out[i][0] *= 2.0f;
            fft_out[i][1] *= 2.0f;
        } else {
            fft_out[i][0] = 0.0f;
            fft_out[i][1] = 0.0f;
        }
    }

    // Perform inverse FFT
    fftwf_plan iplan = fftwf_plan_dft_1d(n, fft_out, fft_in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(iplan);
    fftwf_destroy_plan(iplan);

    // Construct analytic signal: original + j*hilbert_transform
    // Note: IFFT result needs to be normalized by N
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::complex<float>(
            in[i],              // Original signal
            fft_in[i][1] / n    // Hilbert transform (normalized IFFT)
        );
    }

    // Clean up FFTW resources
    fftwf_free(fft_in);
    fftwf_free(fft_out);
}

py::array_t<float> cal_hilbert_phase( py::array_t<float> input)  {
    pybind11::buffer_info buf_input = input.request();
    float* input_ptr = static_cast<float *>(buf_input.ptr);
    size_t N = buf_input.shape[0];
    std::complex<float>  output[N];
    for (size_t i=0; i < N; ++i){
        output[i] = std::complex<float>(0.0f, 0.0f);
    }
    hilbert_transform(input_ptr,output,N);
    std::vector<size_t> hilbert_shape = {N};
    py::array_t hilbert_phase_array = py::array_t<float>(hilbert_shape);
    pybind11::buffer_info buf_hilbert = hilbert_phase_array.request();
    float * buf_hilbert_ptr = static_cast<float *>(buf_hilbert.ptr);
    for (size_t i = 0; i < N; ++i){
        buf_hilbert_ptr[i] = std::arg(output[i]);
    }
    return hilbert_phase_array;
}