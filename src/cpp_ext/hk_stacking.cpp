#include "hk_stacking.h"
#include "mathfuns.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <complex>
#include <cmath>
#define SUPRESS 10



//  constructor
Hk_stacking_multilayer_Vp::Hk_stacking_multilayer_Vp(
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
) : nlayer(nlayer), 
    delta(delta),
    ray_params(ray_params),
    Vps(Vps),
    Hs(Hs),
    Ks(Ks),
    phasesize(phasesize),
    phaselist(phaselist),
    alphas(alphas),
    traces_suppress(traces_suppress),
    input_data(stack_data)

{
    // std::cout << "Hk_stacking_multilayer_Vp created." << std::endl;
}

// destructor
Hk_stacking_multilayer_Vp::~Hk_stacking_multilayer_Vp() {
    // std::cout << "Hk_stacking_multilayer_Vp destroyed." << std::endl;
}


// initialize function
void Hk_stacking_multilayer_Vp::check_initialize() {
    // This function initializes the class, if needed
    // For now, we will just print a message
    // std::cout << "Hk_stacking_multilayer_Vp initialized." << std::endl;
    // check parameters
    // check if the sizes of vectors match the number of layers
    if (Vps.size()!= nlayer || Hs.size()!= 2*nlayer+1 || Ks.size() != 2*nlayer+1 || phasesize.size() != nlayer) {
        throw std::runtime_error("The vector, for Vps, Hs, Ks or phasesize, does not match the number of layers.");
    }
    // check the stack data's dimension
    pybind11::buffer_info buf_input = input_data.request();
    if (buf_input.shape.size() != 3) {
        throw std::runtime_error("The dimension of stack data does not match 3.");
    }
    //initialize the input_data_ptr and shape variables
    input_data_ptr = static_cast<float *>(buf_input.ptr);
    I1 = buf_input.shape[0];  // Size along the first dimension (trace)
    I2 = buf_input.shape[1];  // Size along the second dimension (ray parameter)
    I3 = buf_input.shape[2];  // Size along the third dimension (time)

    if (I1!=alphas.size()) {
        throw std::runtime_error("The 1th dimension of stack data does not match the size of alphas");
    }

    if (I2!=ray_params.size()) {
        throw std::runtime_error("The 2th dimension of stack data does not match the size of ray_params");
    }

    // check if the size of phaselist matches the total size of phasesize
    // check the trace index
    int current_index = 0;

    for (size_t i = 0; i < nlayer; ++i) {
        if (phasesize[i] <= 0) {
            throw std::runtime_error("Parameter size for layer " + std::to_string(i) + " is not positive.");
        }
        for (int j = 0; j < phasesize[i]; ++j){
            int trace_idx = current_index + j*(2 * (i + 2))+1;
            // std::cout << trace_idx << "  " << phaselist[trace_idx] << std::endl ;
            if (trace_idx >= phaselist.size()) {
                throw std::runtime_error("Total parameter size does not match the size of param_value vector.");
            }
            if (phaselist[trace_idx] >= I1 || phaselist[trace_idx] < 0) {

                throw std::runtime_error("Trace index for layer " + std::to_string(i) + " phase " + std::to_string(j) + " is out of bounds.");
            }
        }
        current_index += phasesize[i] * (2 * (i + 2));
    }
    if (current_index != phaselist.size()) {
        throw std::runtime_error("Total parameter size does not match the size of param_value vector.");
    }

    initphaseidx_layers.resize(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        if (i==0) {
            initphaseidx_layers[i] = 0;
        }else{
            initphaseidx_layers[i] = phasesize[i-1]*(i+1)*2 + initphaseidx_layers[i-1];
        }
        
    }
    size_t nphase = 0;
    for (size_t i=0; i<nlayer; ++i){
        nphase += phasesize[i];
    }


    //initialize the phasetimes
    std::vector<size_t> phasetimes_shape = {nphase, I2};
    phasetimes = py::array_t<float>(phasetimes_shape);
    pybind11::buffer_info buf_phasetimes = phasetimes.request();
    phasetimes_ptr = static_cast<float *>(buf_phasetimes.ptr);
    P1 = buf_phasetimes.shape[0];
    P2 = buf_phasetimes.shape[1];
    for (size_t i=0;i<P1*P2;++i){
        phasetimes_ptr[i] = 0.0f;
    }


    // initialize the Hs_optimal and Ks_optimal vectors
    Hs_optimal.resize(nlayer);
    Ks_optimal.resize(nlayer);
    for (size_t i = 0; i < nlayer; ++i) {
        Hs_optimal[i] = Hs[2*i+1];  // Initialize with the first value of Hs for each layer
        Ks_optimal[i] = Ks[2*i+1];  // Initialize with the first value of Ks for each layer
    }


    // initialize the stacked image as a py::array_t<float>
    nH = static_cast<size_t>(Hs[0]);
    nK = static_cast<size_t>(Ks[0]);
    std::vector<size_t> stacked_shape = {nlayer, nH, nK};
    stacked_image = py::array_t<float>(stacked_shape);
    pybind11::buffer_info buf_image = stacked_image.request();
    stacked_image_ptr = static_cast<float *>(buf_image.ptr);
    S1 = buf_image.shape[0];  // Size along the first dimension (layer)
    S2 = buf_image.shape[1];  // Size along the second dimension (nH)
    S3 = buf_image.shape[2];  // Size along the third dimension (nK)
    for (size_t i = 0; i < S1 * S2 * S3; ++i) {
        stacked_image_ptr[i] = 0.0f;  // Initialize all values to zero
    }




}


// Hk_stacking function
void Hk_stacking_multilayer_Vp::Hk_stacking() {
    // This function performs the Hk stacking operation and returns a NumPy array
    // For now, we will just return the stack_data as a placeholder
    // In a real implementation, you would perform the Hk stacking operation here
    // std::cout << "Hk_stacking_multilayer_Vp Hk_stacking called." << std::endl;
    check_initialize();  // Ensure the class is initialized before proceeding
    
    // Perform the Hk stacking operation
    // This is a placeholder for the actual Hk stacking logic
    size_t phasecount = 0;
    for (size_t layer = 0; layer < nlayer; ++layer) {
        int nphase = phasesize[layer];
        float dH = (Hs[2*layer+2] - Hs[2*layer+1])/ static_cast<float>(nH-1);
        float H1 = Hs[2*layer+1];
        float dK = (Ks[2*layer+2] - Ks[2*layer+1])/ static_cast<float>(nK-1);
        float K1 = Ks[2*layer+1];
        // std::cerr << "Layer: " << layer << ", Vp: " << Vps[layer] << ", dH: " << dH 
        //           << ", H1: " << H1 << ", dK: " << dK << ", K1: " << K1 << ", Hs_optimal_guass:" << Hs_optimal[layer]
        //            << ", Ks_optimal_guass:" << Ks_optimal[layer] << "\n";

        //stacking for each phase
        for (int phase = 0; phase < nphase; ++phase) {
            int trace_idx = initphaseidx_layers[layer] + phase*2*(layer + 2) + 1; // Calculate the trace index
            int trace = phaselist[trace_idx]; // Get the trace index from the phaselist
            int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ; // Get the weight for this phase
            float weight = phaselist[weight_idx];
            std::vector<int> phase_P(layer+1);
            std::vector<int> phase_S(layer+1);
            // std::cerr <<weight <<" ";
            // std::cerr <<trace<<" ";            
            for (size_t i = 0; i < layer+1; ++i) {
                phase_S[i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+2]; 
                phase_P[i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+3];
                // std::cerr <<phase_S[i]<<" "<<phase_P[i]<<" ";   

            }
            // std::cerr <<"\n";

            // continue;
            //stacking at each H and K
            for (size_t h = 0; h < nH; ++h) {
                float h_now = H1 + h * dH;  // Current H value
                for (size_t k = 0; k < nK; ++k) {
                    float k_now = K1 + k * dK;  // Current K value
                    // calculate the synthetic time lags for this phase 
                    for (size_t p = 0; p < ray_params.size(); ++p) {
                        float syn_time_lag = h_now * (phase_S[layer]*sqrtf32(squaref32(k_now)/squaref32(Vps[layer])-squaref32(ray_params[p])) + 
                                             phase_P[layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                        for (size_t i = 0; i < layer; ++i) {
                            syn_time_lag += Hs_optimal[i]*(phase_S[i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) +
                                            phase_P[i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                        }
                        size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);

                        // // std::cerr << "h: " << h_now << ", k: " << k_now << ", syn_time_lag: " << syn_time_lag 
                        // //           << ", syn_time_lag_idx: " << syn_time_lag_idx << "\n";
                        size_t data_idx = I3*I2*trace + I3*p + syn_time_lag_idx;
                        // Check if the index is within bounds  
                        if (data_idx >= I1 * I2 * I3) {
                            // std::cerr  << " syn_time_lag " << syn_time_lag << ". Skipping this iteration.\n";
                            std::cerr << h_now << " " << k_now << " " << syn_time_lag << " " << syn_time_lag_idx << " " << ray_params[p] <<"\n";
                            // std::cerr << "Index out of bounds: " << data_idx << " >= " << I1 * I2 * I3 << "\n";
                            exit(1);  // Skip this iteration if index is out of bounds
                        }
                        stacked_image_ptr[S3*S2*layer + S3*h + k] += weight*input_data_ptr[data_idx];
                            
                    }
                }
            }
        }

        //grid searching for the optimal Hs and Ks
        float amp_max = 0.0f;
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK;  // Current K value
                if (stacked_image_ptr[S3*S2*layer + S3*h + k] > amp_max) {
                    amp_max = stacked_image_ptr[S3*S2*layer + S3*h + k];
                    // save the optimal Hs and Ks
                    Hs_optimal[layer] = h_now;
                    Ks_optimal[layer] = k_now;
                }   
            
            }
        }

        for (int phase = 0; phase < nphase; ++phase) {
            std::vector<int> phase_P(layer+1);
            std::vector<int> phase_S(layer+1);
            for (size_t i = 0; i < layer+1; ++i) {
                phase_S[i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+2]; 
                phase_P[i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+3];
                // std::cerr <<phase_S[i]<<" "<<phase_P[i]<<" ";   

            }
            //int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2);
            //float weight_sgn = sgn(phaselist[weight_idx]);
            for (size_t p = 0; p < ray_params.size(); ++p) {
                ///////////////////////////////////////////////////////////////////////////////
                // calculate the synthetic time lags for each phase  and each ray parameter
                float syn_time_lag = Hs_optimal[layer] * (phase_S[layer]*sqrtf32(squaref32(Ks_optimal[layer])/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_P[layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));

                for (size_t i = 0; i < layer; ++i) {
                    syn_time_lag += Hs_optimal[i] * (phase_S[i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_P[i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                }
                phasetimes_ptr[(phasecount+phase)*P2+p] = syn_time_lag;

            }

        }
        phasecount += nphase;




    }

}


// Hk_stacking function
void Hk_stacking_multilayer_Vp::Hk_stacking_PWS() {
    // This function performs the Hk stacking operation and returns a NumPy array
    // For now, we will just return the stack_data as a placeholder
    // In a real implementation, you would perform the Hk stacking operation here
    check_initialize();  // Ensure the class is initialized before proceeding
    
    //calculate the Hilbert transform of each traces
    std::complex<float>* input_hilbert = new std::complex<float> [I1*I2*I3];
    for (size_t i = 0; i < I1; ++i){
        for (size_t j = 0; j < I2; ++j){
            hilbert_transform(input_data_ptr+I2*I3*i+I3*j,input_hilbert+I2*I3*i+I3*j,I3);
        }
    }
    // Perform the Hk stacking operation
    // This is a placeholder for the actual Hk stacking logic

    for (size_t layer = 0; layer < nlayer; ++layer) {
        int nphase = phasesize[layer];
        float dH = (Hs[2*layer+2] - Hs[2*layer+1])/ static_cast<float>(nH-1);
        float H1 = Hs[2*layer+1];
        float dK = (Ks[2*layer+2] - Ks[2*layer+1])/ static_cast<float>(nK-1);
        float K1 = Ks[2*layer+1];

        //set the parameters for each phase
        std::vector<int> traces (nphase);
        std::vector<std::vector<int>> phase_Ps (nphase,std::vector<int>(layer+1));
        std::vector<std::vector<int>> phase_Ss (nphase,std::vector<int>(layer+1));
        for (int phase = 0; phase < nphase; ++phase) {
            traces[phase] = phaselist[initphaseidx_layers[layer] + phase*2*(layer + 2) + 1]; // Get the trace index from the phaselist  
            for (size_t i = 0; i < layer+1; ++i) {
                phase_Ss[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+2]; 
                phase_Ps[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+3]; 
            }
        }    
    
        //stacking at each H and K
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK; // Current K value
                std::vector<float> sums (nphase, 0.0f);  
                std::vector<float> weights (nphase, 0.0f);// Initialize the weights for the PWD method
                float weight_sum = 0.0f;
                //stacking for each phase            
                for (int phase = 0; phase < nphase; ++phase) {
                    std::complex<float> phi_sum (0.0f, 0.0f) ;
                    for (size_t p = 0; p < ray_params.size(); ++p) {
                        ///////////////////////////////////////////////////////////////////////////////
                        // calculate the synthetic time lags for each phase  and each ray parameter
                        float syn_time_lag = h_now * (phase_Ss[phase][layer]*sqrtf32(squaref32(k_now)/squaref32(Vps[layer])-squaref32(ray_params[p])) + 
                                             phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                        for (size_t i = 0; i < layer; ++i) {
                            syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) +
                                            phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                        }
                        // end of calculate the synthetic time lags for each phase  and each ray parameter
                        ///////////////////////////////////////////////////////////////////////////////
                        size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);
                        if (syn_time_lag_idx >= I3 || syn_time_lag_idx < 0) {
                            // std::cerr  << " syn_time_lag " << syn_time_lag << ". Skipping this iteration.\n";
                            std::cerr << h_now << " " << k_now << " " << syn_time_lag << " " << syn_time_lag_idx << " " << ray_params[p] <<"\n";
                            // std::cerr << "Index out of bounds: " << data_idx << " >= " << I1 * I2 * I3 << "\n";
                            exit(1);  // Skip this iteration if index is out of bounds
                        }
                        size_t data_idx = I3*I2*traces[phase] + I3*p + syn_time_lag_idx;
                        //stack the searched amplitude
                        sums[phase] += input_data_ptr[data_idx];
                        //stack the srarched phase
                        phi_sum += input_hilbert[data_idx]/std::abs(input_hilbert[data_idx]);

                    }
                    weights[phase] = std::abs(phi_sum)/ray_params.size();
                    weight_sum += std::abs(weights[phase]);
                }

                for (int phase = 0; phase < nphase; ++phase) {

                    int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ; // Get the weight for this phase
                    weights[phase] = weights[phase]*sgn(phaselist[weight_idx])/weight_sum;
                    phaselist[weight_idx] = weights[phase];
                    stacked_image_ptr[S3*S2*layer + S3*h + k] += weights[phase]*sums[phase];  // Weighted sum for the CS method
                }
            }
        }


        //grid searching for the optimal Hs and Ks
        float amp_max = 0.0f;
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK;  // Current K value
                if (stacked_image_ptr[S3*S2*layer + S3*h + k] > amp_max) {
                    amp_max = stacked_image_ptr[S3*S2*layer + S3*h + k];
                    // save the optimal Hs and Ks
                    Hs_optimal[layer] = h_now;
                    Ks_optimal[layer] = k_now;
                }   
            }
        }

        //window out the searched phases
        for (int phase = 0; phase < nphase; ++phase) {
            int trace_idx = traces[phase];
            int trace_suppress = traces_suppress[trace_idx];
            if (trace_suppress<0) continue;
            size_t supress_width =  log(10.0)/alphas[trace_suppress]/delta;  //make sure the amplitude suppressed to be its 1/10

            for (size_t p = 0; p < ray_params.size(); ++p) {
                ///////////////////////////////////////////////////////////////////////////////
                // calculate the synthetic time lags for each phase  and each ray parameter
                float syn_time_lag = Hs_optimal[layer] * (phase_Ss[phase][layer]*sqrtf32(squaref32(Ks_optimal[layer])/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));

                for (size_t i = 0; i < layer; ++i) {
                    syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                }
                size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);
                if (syn_time_lag_idx >= I3 || syn_time_lag_idx < 0) {
                    // std::cerr  << " syn_time_lag " << syn_time_lag << ". Skipping this iteration.\n";
                    std::cerr << Hs_optimal[layer] << " " << Ks_optimal[layer] << " " << syn_time_lag << " " << syn_time_lag_idx << " " << ray_params[p] <<"\n";
                    // std::cerr << "Index out of bounds: " << data_idx << " >= " << I1 * I2 * I3 << "\n";
                    exit(1);  // Skip this iteration if index is out of bounds
                }

                //suppress using a box window
                size_t window_bgn = syn_time_lag_idx-supress_width;
                size_t window_end = syn_time_lag_idx+supress_width;
                if (window_bgn<0) window_bgn=0;
                if (window_end>I3-1) window_bgn=I3-1;
                for (size_t pp=window_bgn; pp <= window_end; ++pp){
                    //input_data_ptr[I3*I2*trace_suppress + I3*p + pp] = input_data_ptr[I3*I2*trace_suppress + I3*p + pp]*(1-exp(-1*squaref32(alpha*(pp*delta-syn_time_lag))));
                    input_data_ptr[I3*I2*trace_suppress + I3*p + pp] = 0.0;
                }

            }

        }

    }
    // free the momory
    delete [] input_hilbert;
}


// Hk_stacking function
void Hk_stacking_multilayer_Vp::Hk_stacking_PWS1() {
    // This function performs the Hk stacking operation and returns a NumPy array
    // For now, we will just return the stack_data as a placeholder
    // In a real implementation, you would perform the Hk stacking operation here
    check_initialize();  // Ensure the class is initialized before proceeding
    
    //calculate the Hilbert transform of each traces
    std::complex<float>* input_hilbert = new std::complex<float> [I1*I2*I3];
    for (size_t i = 0; i < I1; ++i){
        for (size_t j = 0; j < I2; ++j){
            hilbert_transform(input_data_ptr+I2*I3*i+I3*j,input_hilbert+I2*I3*i+I3*j,I3);
        }
    }

    // Perform the Hk stacking operation
    // This is a placeholder for the actual Hk stacking logic
    size_t phasecount = 0;
    
    std::vector<int> traces (P1);
    for (size_t layer = 0; layer < nlayer; ++layer) {
        
        int nphase = phasesize[layer];
        float dH = (Hs[2*layer+2] - Hs[2*layer+1])/ static_cast<float>(nH-1);
        float H1 = Hs[2*layer+1];
        float dK = (Ks[2*layer+2] - Ks[2*layer+1])/ static_cast<float>(nK-1);
        float K1 = Ks[2*layer+1];

        //set the parameters for each phase
        
        std::vector<std::vector<int>> phase_Ps (nphase,std::vector<int>(layer+1));
        std::vector<std::vector<int>> phase_Ss (nphase,std::vector<int>(layer+1));
        for (int phase = 0; phase < nphase; ++phase) {
            //std::cerr << layer << " " << phase << " " << initphaseidx_layers[layer] << " " << phaselist[initphaseidx_layers[layer] + phase*2*(layer + 2) + 1] << std::endl; 
            traces[phasecount+phase] = phaselist[initphaseidx_layers[layer] + phase*2*(layer + 2) + 1];
            for (size_t i = 0; i < layer+1; ++i) {
                phase_Ss[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+2]; 
                phase_Ps[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+3]; 
            }
        }
        //std::cerr << layer << " " << nphase << std::endl;    
    
        //stacking at each H and K
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK; // Current K value
                std::vector<float> sums (nphase, 0.0f);  
                std::vector<float> weights (nphase, 0.0f);// Initialize the weights for the PWD method
                float weight_sum = 0.0f;
                //stacking for each phase            
                for (int phase = 0; phase < nphase; ++phase) {
                    std::complex<float> phi_sum (0.0f, 0.0f) ;
                     
                    //int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2);
                    //float weight_sgn = sgn(phaselist[weight_idx]);
                    for (size_t p = 0; p < ray_params.size(); ++p) {
                        
                        ///////////////////////////////////////////////////////////////////////////////
                        // calculate the synthetic time lags for each phase  and each ray parameter
                        float syn_time_lag = h_now * (phase_Ss[phase][layer]*sqrtf32(squaref32(k_now)/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                        for (size_t i = 0; i < layer; ++i) {
                            syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                        }
                        // end of calculate the synthetic time lags for each phase  and each ray parameter
                        ///////////////////////////////////////////////////////////////////////////////
                        size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);
                        if (syn_time_lag_idx >= I3 || syn_time_lag_idx < 0) {
                            // std::cerr  << " syn_time_lag " << syn_time_lag << ". Skipping this iteration.\n";
                            std::cerr << h_now << " " << k_now << " " << syn_time_lag << " " << syn_time_lag_idx << " " << ray_params[p] <<"\n";
                            // std::cerr << "Index out of bounds: " << data_idx << " >= " << I1 * I2 * I3 << "\n";
                            exit(1);  // Skip this iteration if index is out of bounds
                        } 
                        int trace_num = traces[phasecount+phase];
                        size_t data_idx = I3*I2*trace_num + I3*p + syn_time_lag_idx;
                        // if (layer==2 ){
                        //      std::cerr << layer << " " << nphase << " " << phase <<" " << trace_num << " " << data_idx << " " << I1*I2*I3 << std::endl;   

                        // }

                        //-------------------------------------------------------------------------------------
                        //searching the close searched phase and suppress its weight
                        float alpha = alphas[trace_num];
                        float T_cos = 8*log(SUPRESS)/alpha;
                        float weight_suppress = 1.0f;
                        for (size_t i=0; i<phasecount; ++i){
                            int trace_num1 = traces[i];
                            
                            
                            // float alpha = 0.1;
                            if (traces_suppress[trace_num]==traces_suppress[trace_num1]){
                                
                                //weight_suppress = weight_suppress*(1-exp(-squaref32(alpha/3.0*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                                // weight_suppress = weight_suppress*pow((1-exp(-squaref32(alpha*(syn_time_lag-phasetimes_ptr[i*P2+p])))),2);
                                if (abs(syn_time_lag-phasetimes_ptr[i*P2+p])<2.*sqrtf32(log(SUPRESS))/alpha) 
                                {
                                    //weight_suppress = weight_suppress*0.01;
                                    weight_suppress = weight_suppress*(1-squaref32(std::cos(2*M_PI/T_cos*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                                    }

                            }
                        }

                        //suppress this phase if observed polarity doesn't match its true polarity. 
                        //if (weight_sgn*input_data_ptr[data_idx]<0) weight_suppress = weight_suppress*0.01;
                        
                        //stack the searched amplitude
                        sums[phase] += input_data_ptr[data_idx];
                        //stack the searched phase weight            
                        phi_sum += input_hilbert[data_idx]/std::abs(input_hilbert[data_idx])*weight_suppress; 

                    }
                    weights[phase] = pow(std::abs(phi_sum)/ray_params.size(),2);
                    weight_sum += weights[phase];
                }

                for (int phase = 0; phase < nphase; ++phase) {

                    int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2); // Get the weight for this phase
                    weights[phase] = weights[phase]*sgn(phaselist[weight_idx])/(weight_sum+0.001);
                    stacked_image_ptr[S3*S2*layer + S3*h + k] += weights[phase]*sums[phase];  // Weighted sum for the CS method
                }
            }
        }


        //grid searching for the optimal Hs and Ks
        float amp_max = 0.0f;
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK;  // Current K value
                if (stacked_image_ptr[S3*S2*layer + S3*h + k] > amp_max) {
                    amp_max = stacked_image_ptr[S3*S2*layer + S3*h + k];
                    // save the optimal Hs and Ks
                    Hs_optimal[layer] = h_now;
                    Ks_optimal[layer] = k_now;
                }
            }
        }




        //std::cerr << 111 << std::endl;
        std::vector<float> sums1 (nphase, 0.0f);  
        std::vector<float> weights1 (nphase, 0.0f);// Initialize the weights for the PWD method
        float weight_sum1 = 0.0f;
        //store the searched phasetimes and weights
        for (int phase = 0; phase < nphase; ++phase) {
            //std::cerr << phase << std::endl;
            std::complex<float> phi_sum (0.0f, 0.0f) ;
            //int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2);
            //float weight_sgn = sgn(phaselist[weight_idx]);
            for (size_t p = 0; p < ray_params.size(); ++p) {
                ///////////////////////////////////////////////////////////////////////////////
                // calculate the synthetic time lags for each phase  and each ray parameter
                float syn_time_lag = Hs_optimal[layer] * (phase_Ss[phase][layer]*sqrtf32(squaref32(Ks_optimal[layer])/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));

                for (size_t i = 0; i < layer; ++i) {
                    syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                }
                phasetimes_ptr[(phasecount+phase)*P2+p] = syn_time_lag;
                size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);
                //std::cout << syn_time_lag << std::endl;

                int trace_num = traces[phasecount+phase];
                size_t data_idx = I3*I2*trace_num + I3*p + syn_time_lag_idx;
                //stack the searched amplitude
                float weight_suppress = 1.0f;
                float alpha = alphas[trace_num];
                float T_cos = 8*log(SUPRESS)/alpha;
                for (size_t i=0; i<phasecount; ++i){
                    int trace_num1 = traces[i];
                    // float alpha = 0.1;
                    if (traces_suppress[trace_num]==traces_suppress[trace_num1]){
                        
                        // std::cout<<trace_num<<" "<<i<<std::endl;
                        //weight_suppress = weight_suppress*(1-exp(-squaref32(alpha/3.0*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                        if (abs(syn_time_lag-phasetimes_ptr[i*P2+p])<2.*sqrtf32(log(SUPRESS))/alpha) {
                            //weight_suppress = weight_suppress*0.01;
                            //std::cerr <<"layer"<<layer<< ":" << "phase1 " << phase +phasecount<< "," << "phase2 " << i << std::endl;
                            weight_suppress = weight_suppress*(1-squaref32(std::cos(2*M_PI/T_cos*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                        }

                    }
                }
                // if (layer==1 && phase==0) std::cerr << weight_suppress << std::endl;
                //if (weight_sgn*input_data_ptr[data_idx]<0) weight_suppress = weight_suppress*0.01;
                //if (layer==1 && phase==1) std::cout<<weight_sgn<<" "<<input_data_ptr[data_idx]<<std::endl;
                phi_sum += input_hilbert[data_idx]/std::abs(input_hilbert[data_idx])*weight_suppress; 

            }
            

            weights1[phase] = pow(std::abs(phi_sum)/ray_params.size(),2);
            //std::cerr <<"phase " << phase << ":" << weights1[phase] << std::endl;
            weight_sum1 += weights1[phase];

        }

       //std::cerr << 222 << std::endl;
        for (int phase = 0; phase < nphase; ++phase) {
            int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ; // Get the weight for this phase
            weights1[phase] = weights1[phase]*sgn(phaselist[weight_idx])/weight_sum1;
            //std::cout<<"phase"<<phase<<" "<<weights1[phase]<<std::endl;
            phaselist[weight_idx] = weights1[phase];

        }
        phasecount += nphase;
    }
    // free the momory
    delete [] input_hilbert;
}


// Hk_stacking function
void Hk_stacking_multilayer_Vp::Hk_stacking_PWS2() {
    // This function performs the Hk stacking operation and returns a NumPy array
    // For now, we will just return the stack_data as a placeholder
    // In a real implementation, you would perform the Hk stacking operation here
    check_initialize();  // Ensure the class is initialized before proceeding
    
    //calculate the Hilbert transform of each traces
    std::complex<float>* input_hilbert = new std::complex<float> [I1*I2*I3];
    for (size_t i = 0; i < I1; ++i){
        for (size_t j = 0; j < I2; ++j){
            hilbert_transform(input_data_ptr+I2*I3*i+I3*j,input_hilbert+I2*I3*i+I3*j,I3);
        }
    }

    // Perform the Hk stacking operation
    // This is a placeholder for the actual Hk stacking logic
    size_t phasecount = 0;
    
    std::vector<int> traces (P1);
    for (size_t layer = 0; layer < nlayer; ++layer) {
        
        int nphase = phasesize[layer];
        float dH = (Hs[2*layer+2] - Hs[2*layer+1])/ static_cast<float>(nH-1);
        float H1 = Hs[2*layer+1];
        float dK = (Ks[2*layer+2] - Ks[2*layer+1])/ static_cast<float>(nK-1);
        float K1 = Ks[2*layer+1];

        //set the parameters for each phase

        std::vector<std::vector<int>> phase_Ps (nphase,std::vector<int>(layer+1));
        std::vector<std::vector<int>> phase_Ss (nphase,std::vector<int>(layer+1));
        for (int phase = 0; phase < nphase; ++phase) {
            //std::cerr << layer << " " << phase << " " << initphaseidx_layers[layer] << " " << phaselist[initphaseidx_layers[layer] + phase*2*(layer + 2) + 1] << std::endl; 
            traces[phasecount+phase] = phaselist[initphaseidx_layers[layer] + phase*2*(layer + 2) + 1];
            for (size_t i = 0; i < layer+1; ++i) {
                phase_Ss[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+2]; 
                phase_Ps[phase][i] = phaselist[initphaseidx_layers[layer]+ phase*2*(layer + 2) + 2*i+3]; 
            }
        }
        //std::cerr << layer << " " << nphase << std::endl;
        std::vector<std::vector<std::vector<float>>> phase_sums ;
        phase_sums.resize(nphase);
        for (size_t phase = 0; phase < nphase; ++phase){
            phase_sums[phase].resize(nH);
            for (size_t h=0; h < nH; ++h){
                phase_sums[phase][h].resize(nK,0);
            }
        }
    
        //stacking at each H and K
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK; // Current K value
                std::vector<float> sums (nphase, 0.0f);  
                std::vector<float> weights (nphase, 0.0f);// Initialize the weights for the PWD method

                //stacking for each phase            
                for (int phase = 0; phase < nphase; ++phase) {
                    std::complex<float> phi_sum (0.0f, 0.0f) ;
                     
                    //int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2);
                    //float weight_sgn = sgn(phaselist[weight_idx]);
                    for (size_t p = 0; p < ray_params.size(); ++p) {
                        
                        ///////////////////////////////////////////////////////////////////////////////
                        // calculate the synthetic time lags for each phase  and each ray parameter
                        float syn_time_lag = h_now * (phase_Ss[phase][layer]*sqrtf32(squaref32(k_now)/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                        for (size_t i = 0; i < layer; ++i) {
                            syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                        }
                        // end of calculate the synthetic time lags for each phase  and each ray parameter
                        ///////////////////////////////////////////////////////////////////////////////
                        size_t syn_time_lag_idx = static_cast<size_t>(syn_time_lag / delta);
                        if (syn_time_lag_idx >= I3 || syn_time_lag_idx < 0) {
                            // std::cerr  << " syn_time_lag " << syn_time_lag << ". Skipping this iteration.\n";
                            std::cerr << h_now << " " << k_now << " " << syn_time_lag << " " << syn_time_lag_idx << " " << ray_params[p] <<"\n";
                            // std::cerr << "Index out of bounds: " << data_idx << " >= " << I1 * I2 * I3 << "\n";
                            exit(1);  // Skip this iteration if index is out of bounds
                        }
                        int trace_num = traces[phasecount+phase];
                        size_t data_idx = I3*I2*trace_num + I3*p + syn_time_lag_idx;

                        //stack the searched amplitude
                        sums[phase] += input_data_ptr[data_idx];
                        //stack the searched phase weight            
                        phi_sum += input_hilbert[data_idx]/std::abs(input_hilbert[data_idx]); 

                    }
                    weights[phase] = pow(std::abs(phi_sum)/ray_params.size(),2);
                    sums[phase] = sums[phase]/ray_params.size();
                }

                for (int phase = 0; phase < nphase; ++phase) {
                    int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2); // Get the weight for this phase
                    weights[phase] = weights[phase]*sgn(phaselist[weight_idx]);
                    float sum_temp = weights[phase]*sums[phase];
                    if (sum_temp<0) sum_temp=0;
                    phase_sums[phase][h][k] = phase_sums[phase][h][k] + sum_temp;
                    //stacked_image_ptr[S3*S2*layer + S3*h + k] += weights[phase]*sums[phase];  // Weighted sum for the CS method
                }
            }
        }

        //normalize each phase_sums[phase] by the its max
        for (int phase = 0; phase < nphase; ++phase) {
            float each_phase_sum_amp = 0.0f;

            for (size_t h = 0; h < nH; ++h) {
                //float h_now = H1 + h * dH;  // Current H value
                for (size_t k = 0; k < nK; ++k) {
                    //float k_now = K1 + k * dK; // Current K value
                    if (phase_sums[phase][h][k]>each_phase_sum_amp){
                        each_phase_sum_amp = phase_sums[phase][h][k];
                    }
                }
            }

            if (each_phase_sum_amp <= 0.0f) {
                each_phase_sum_amp = 1.0f;
            }

            //stack the normalized phase_sums[phase]
            for (size_t h = 0; h < nH; ++h) {

                float h_now = H1 + h * dH;  // Current H value
                for (size_t k = 0; k < nK; ++k) {
                    float k_now = K1 + k * dK; // Current K value

                    //-------------------------------------------------------------------------------------
                    //searching the close searched phase and suppress its weight
                    int trace_num = traces[phasecount+phase];
                    float alpha = alphas[trace_num];
                    float T_cos = 8*log(SUPRESS)/alpha;
                    float weight_suppress = 1.0f;
                    for (size_t i=0; i<phasecount; ++i){
                        int trace_num1 = traces[i];
                        // float alpha = 0.1;
                        if (traces_suppress[trace_num]==traces_suppress[trace_num1]){
                            float t_phase1 = 0;
                            size_t p = 0;
                            float syn_time_lag = h_now * (phase_Ss[phase][layer]*sqrtf32(squaref32(k_now)/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                            for (size_t i = 0; i < layer; ++i) {
                                syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                            }
                            
                            //weight_suppress = weight_suppress*(1-exp(-squaref32(alpha/3.0*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                            // weight_suppress = weight_suppress*pow((1-exp(-squaref32(alpha*(syn_time_lag-phasetimes_ptr[i*P2+p])))),2);
                            if (abs(syn_time_lag-phasetimes_ptr[i*P2+p])<2.*sqrtf32(log(SUPRESS))/alpha) 
                            {
                                //weight_suppress = weight_suppress*0.01;
                                weight_suppress = weight_suppress*(1-squaref32(std::cos(2*M_PI/T_cos*(syn_time_lag-phasetimes_ptr[i*P2+p]))));
                            }
                        }
                    }
                    stacked_image_ptr[S3*S2*layer + S3*h + k] += phase_sums[phase][h][k]/each_phase_sum_amp*weight_suppress;
                    phase_sums[phase][h][k] = phase_sums[phase][h][k]/each_phase_sum_amp;
                }
            }

        }
        
        //grid searching for the optimal Hs and Ks
        float amp_max = 0.0f;
        size_t h_opt = 0;
        size_t k_opt = 0;
        for (size_t h = 0; h < nH; ++h) {
            float h_now = H1 + h * dH;  // Current H value
            for (size_t k = 0; k < nK; ++k) {
                float k_now = K1 + k * dK;  // Current K value
                if (stacked_image_ptr[S3*S2*layer + S3*h + k] > amp_max) {
                    amp_max = stacked_image_ptr[S3*S2*layer + S3*h + k];
                    // save the optimal Hs and Ks
                    Hs_optimal[layer] = h_now;
                    Ks_optimal[layer] = k_now;
                    h_opt = h;
                    k_opt = k;
                }
            }
        }


       //std::cerr << 222 << std::endl;
        for (int phase = 0; phase < nphase; ++phase) {
            int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ; // Get the weight for this phase
            //std::cout<<"phase"<<phase<<" "<<weights1[phase]<<std::endl;
            phaselist[weight_idx] = phase_sums[phase][h_opt][k_opt];
            for (size_t p = 0; p < ray_params.size(); ++p) {
                ///////////////////////////////////////////////////////////////////////////////
                // calculate the synthetic time lags for each phase  and each ray parameter
                float syn_time_lag = Hs_optimal[layer] * (phase_Ss[phase][layer]*sqrtf32(squaref32(Ks_optimal[layer])/squaref32(Vps[layer])-squaref32(ray_params[p])) + phase_Ps[phase][layer]*sqrtf32(1.0/squaref32(Vps[layer])-squaref32(ray_params[p])));
                for (size_t i = 0; i < layer; ++i) {
                    syn_time_lag += Hs_optimal[i] * (phase_Ss[phase][i]*sqrtf32(squaref32(Ks_optimal[i])/squaref32(Vps[i])-squaref32(ray_params[p])) + phase_Ps[phase][i]*sqrtf32(1.0/squaref32(Vps[i])-squaref32(ray_params[p])));
                }
                phasetimes_ptr[(phasecount+phase)*P2+p] = syn_time_lag;
            }
        }
        phasecount += nphase;
    }
    // free the momory
    delete [] input_hilbert;
}






// test the constuctor
void Hk_stacking_multilayer_Vp::test_constructor() {
    check_initialize();
    std::cout << "Hk_stacking_multilayer_Vp constructor called." << std::endl;
    std::cout << "Number of layers: " << nlayer << std::endl;
    std::cout << "Delta: " << delta << std::endl;
    std::cout << "Ray parameters size: " << ray_params.size() << std::endl;
    std::cout << "Vps size: " << Vps.size() << std::endl;
    std::cout << "Hs size: " << Hs.size() << std::endl;
    std::cout << "Ks size: " << Ks.size() << std::endl;
    std::cout << "Phasesize size: " << phasesize.size() << std::endl;
    std::cout << "Phaselist size: " << phaselist.size() << std::endl;
    std::cout << "Stack data size: " << I1 << " x " << I2 << " x " << I3 << std::endl;
}


py::array_t<float> Hk_stacking_multilayer_Vp::get_stacked_image() const {
    return stacked_image;
}

py::array_t<float> Hk_stacking_multilayer_Vp::get_phasetimes() const {
    return phasetimes;
}

std::vector <float> Hk_stacking_multilayer_Vp::get_Hs_optimal() const {
    return Hs_optimal;
} 

std::vector <float> Hk_stacking_multilayer_Vp::get_Ks_optimal() const {
    return Ks_optimal;
} 

std::vector <float> Hk_stacking_multilayer_Vp::get_Hs() const {
    return Hs;
}
std::vector <float> Hk_stacking_multilayer_Vp::get_Ks() const {
    return Ks;
}
std::vector <float> Hk_stacking_multilayer_Vp::get_Vps() const {
    return Vps;
}

std::vector <float> Hk_stacking_multilayer_Vp::get_weights() const {
    std::vector<float> weights(P1);
    int phasecount = 0;
    for (size_t layer = 0; layer < nlayer; ++layer) {
        int nphase = phasesize[layer];
        for (int phase = 0; phase < nphase; ++phase) {
            int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ;
            weights[phasecount+phase] = phaselist[weight_idx];
        }
        phasecount += nphase;

    }
    return weights;
}


void Hk_stacking_multilayer_Vp::print_weights() const{
    for (size_t layer = 0; layer < nlayer; ++layer) {
        int nphase = phasesize[layer];
        std::cout<<"Layer: "<<layer << std::endl;
        for (int phase = 0; phase < nphase; ++phase) {
            int weight_idx = initphaseidx_layers[layer]+ phase*2*(layer + 2) ;
            std::cout<<"     phase: " << phase << "   weight: " << phaselist[weight_idx] << std::endl;
        }
    }
}


py::array_t<float> Hk_stacking_multilayer_Vp::cal_hilbert_phase( py::array_t<float> input) const {
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