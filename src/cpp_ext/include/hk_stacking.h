#pragma once
#include <vector>
#include <pybind11/numpy.h> 
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>  
#include <complex>
namespace py = pybind11;

class Hk_stacking_multilayer_Vp {
public:
    // delares the constructor：initializes the class with necessary parameters
    Hk_stacking_multilayer_Vp(
        const size_t& nlayer,
        const float& delta,
        const std::vector<float>& ray_params,
        const std::vector<float>& Vps,
        const std::vector<float>& Hs,
        const std::vector<float>& Ks,
        const std::vector<int>& phasesize,
        const std::vector<float>& phaselist,
        const std::vector<float>& alphas,
        const std::vector<int>& traces_suppress,
        py::array_t<float>& stack_data

    );

    // delares the destructor: cleans up resources
    ~Hk_stacking_multilayer_Vp();

    void check_initialize();

    // delares the Hk_stacking function
    // This function performs the Hk stacking operation and returns a NumPy array
    void Hk_stacking();
    void Hk_stacking_PWS(); //PWS stands for Phase-weighted Stacks
    void Hk_stacking_PWS1(); //PWS stands for Phase-weighted Stacks, 1 represents suppress the sedimentary multiple phases
    void Hk_stacking_PWS2(); //PWS stands for Phase-weighted Stacks; 2 represents searching twice: the first finds the optimal weights; the second determines the optimal H and k
    //functions to get the results
    py::array_t<float> get_stacked_image() const;
    py::array_t<float> get_phasetimes() const;
    py::array_t<float> cal_hilbert_phase(py::array_t<float> input) const;
    std::vector <float> get_Hs_optimal() const;
    std::vector <float> get_Ks_optimal() const;
    std::vector <float> get_Hs() const;
    std::vector <float> get_Ks() const;
    std::vector <float> get_Vps() const;
    std::vector <float> get_weights() const;
    void  print_weights() const;

    // test the constructor 
    void test_constructor();


private:
    size_t nlayer;  // number of layers in the model
    size_t nH;  // number of Hs in the model
    size_t nK;  // number of Ks in the model

    float delta;  // time interval for stacking
    std::vector<float> ray_params;  // ray parameters for each teleseismic event
    std::vector<float> Vps;         // Asummed Vp values in each layer
    std::vector<float> Hs;          // Searching thickness range in each layer
    std::vector<float> Ks;          // Searching Vp/Vs ratio range in each layer
    std::vector<int> phasesize;     // The number of phases in each layer
    std::vector<float> phaselist;   // All the phases used to do Hk stacking
    std::vector<int> initphaseidx_layers;
    std::vector<float> alphas;      // The gaussian parameter for each trace
    std::vector<int> traces_suppress; //the trace number where the phases trace should be suppressed; The traces should have the same data sourcewith their index trace
    

    py::array_t<float> input_data;  // Numpy array to save the RF and AC trace
    float * input_data_ptr;  // Pointer to the data in the NumPy array for direct access
    size_t I1;  // Size of the NumPy array along the first dimension (trace) for bounds checking
    size_t I2;  // Size of the NumPy array along the second dimension (ray paramter) for bounds checking
    size_t I3;  // Size of the NumPy array along the third dimension (time) for bounds checking
    
    
    py::array_t<float> stacked_image;  // Numpy array to save the stacked image
    float * stacked_image_ptr;  // Pointer to the data in the stacked image for direct access
    size_t S1;  // Size of the stacked image along the first dimension (layer) for bounds checking
    size_t S2;  // Size of the stacked image along the second dimension (nH) for bounds checking
    size_t S3;  // Size of the stacked image along the third dimension (nK) for bounds checking

    std::vector<float> Hs_optimal;
    std::vector<float> Ks_optimal;

    py::array_t<float> phasetimes;
    float * phasetimes_ptr;  
    size_t P1;
    size_t P2;


};