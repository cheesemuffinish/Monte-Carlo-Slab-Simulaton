#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define PI 3.14159265
#define MAX 4294967296
#define SIZE 512
using namespace std;

//Generate random numbers
__device__ double rando() {
    int id =  threadIdx.x + blockDim.x*blockIdx.x;
	curandState_t state;
	curand_init(clock64(), id, 0, &state); 
	return (double)curand(&state) / MAX;
    
}

//Initialization of photons
__device__ void new_photon(double *photon){

	double cosTh = rando();
	photon[0] = cosTh;
	double sinTh = sqrt(1.0f - cosTh * cosTh);
	photon[1] = sinTh;
	double phi = 2 * PI * (rando());
	photon[2] = phi;
	double cosPh = cos(phi);
	photon[3] = cosPh;
	double sinPh = sin(phi);
	photon[4] = sinPh;

	//Initializing Position
	double xTotal = 0;
	photon[5] = xTotal;
	double yTotal = 0;
	photon[6] = yTotal;
	double zTotal = 0;
	photon[7] = zTotal;
}

//Isotropic Scattering of photon off of slab
__device__ void iso_scatt(double *photon){
	double cosTh = (rando() * 2 - 1);
	photon[0] = cosTh;
	double sinTh = sqrt(1 - cosTh * cosTh);
	photon[1] = sinTh;
	double phi = 2 * PI * (rando());
	photon[2] = phi;
	double cosPh = cos(phi);
	photon[3] = cosPh;
	double sinPh = sin(phi);
	photon[4] = sinPh;
}


//Moments
__device__ void moments(double x1, double y1, double z1, double x2, double y2, double z2, double cosTh, int nLevel, double *jPlus, double *hPlus, double *kPlus, double *jMinus, double *kMinus, double *hMinus){
	int l1, l2;
	if (z1 > 0 && z2 > 0 && floor(z1 * nLevel) == floor(z2 * nLevel)){
		return;
	}

	if (cosTh > 0){
		if (z1 <= 0){
			l1 = 1;
		}
		else{
			l1 = z1 * nLevel + 2;
		}
		if (z2 >= 1){
			l2 = nLevel + 1;
		}
		else{
			l2 = z2 * nLevel + 1;
		}

		for (int n = l1 - 1; n < l2; n++){
			if (n >= 0) {
				atomicAdd(&jPlus[n], 1.0 / cosTh);
				atomicAdd(&hPlus[n], 1.0);
				atomicAdd(&kPlus[n], cosTh);
			}
			else {
				printf("Error in plus");
			}
			
		}
	}
	else if (cosTh < 0){
		l1 = (z1 * nLevel) + 1;
		if (z2 < 0) {
			l2 = 1;
		}
		else {
			l2 = (z2 * nLevel) + 2;
		}
		for (int n = l2 - 1; n < l1; n++) {
			//__syncthreads();
			if (n >= 0) {
				atomicAdd(&jMinus[n], 1.0 / (abs(cosTh)));
				atomicAdd(&hMinus[n], -1.0);
				atomicAdd(&kMinus[n], abs(cosTh));
			}
			else{
				printf("Error in jMinus");
			}

		}
	}

}

//Generates photons and allows them to propagate from the origin
__global__ void work(double *hPlus, double *kPlus, double *jMinus, double *kMinus, double *hMinus,
	double *jPlus, int muBins, double *energy, int nLevel, double *erri,
	double tauMax, int albedo, int seed, unsigned int *test){
    
    //initialization of photon
	double photon[8] = { 0 };
    
    //atomic adder test
	atomicAdd(&test[0], 1);
    
    //Generate a new photon when it's z2 position is less than 0
    newP:
	new_photon(photon);
	int aFlag = 0;

	while ((photon[7] >= 0) && (photon[7] <= 1)){
		double x1 = photon[5];
		double y1 = photon[6];
		double z1 = photon[7];
		double tau = -log(rando());
		double s = tau / tauMax;
		photon[5] = photon[5] + s*photon[1] * photon[3];
		photon[6] = photon[6] + s*photon[1] * photon[4];
		photon[7] = photon[7] + s*photon[0];
		double x2 = photon[5];
		double y2 = photon[6];
		double z2 = photon[7];
		moments(x1, y1, z1, x2, y2, z2, photon[0], nLevel,
			jPlus, hPlus, kPlus, jMinus, kMinus, hMinus);
		if ((photon[7] < 0) || (photon[7] > 1))
		{
			continue;
		}
		if (rando()< albedo)
		{
			iso_scatt(photon);
		}
		else{
		aFlag = 1;
		continue;
		}
	}
	if (photon[7] < 0){
		goto newP;
	}

	if (aFlag == 0){
		int l = int(muBins*photon[0]);
		if (l >= 0) {
			atomicAdd(&erri[l], 1.0f);
			atomicAdd(&energy[l], 1.0f);
		}
		else {
			printf("Error in erri");
		}
	}
}

//Output 
void output(double hPlus[], double kPlus[], double jMinus[], double kMinus[], double hMinus[], double jPlus[],
	unsigned int nPhotons, int muBins, double intensity[], double energy[], double nLevel, double sigmai[], double theta[], double erri[])
{
	//setting values for arrays
	for (int n = 0; n < muBins; n++){
		intensity[n] = energy[n] / (2 * double(nPhotons)*cos(theta[n] * PI / 180))*muBins;
		sigmai[n] = sqrt(erri[n]) / double(nPhotons);
		//sigmai[n] = erri[n] / nPhotons;
		//sigmai[n] = sqrt(4);
		energy[n] = energy[n] / double(nPhotons);
	}

	for (int n = 0; n < nLevel; n++){
		jPlus[n] = jPlus[n] / double(nPhotons);
		jMinus[n] = jMinus[n] / double(nPhotons);
		hPlus[n] = hPlus[n] / double(nPhotons);
		hMinus[n] = hMinus[n] / double(nPhotons);
		kPlus[n] = kPlus[n] / double(nPhotons);
		kMinus[n] = kMinus[n] / double(nPhotons);
	}
	// write output to file: "intensity.dat"
	ofstream file;
	file.open("intensity.dat");
	for (int i = muBins - 1; i > -1; i--) {
		file << theta[i] << "\t" << energy[i] << "\t" << sigmai[i] << "\t" << intensity[i] << "\n";
	}
	file.close();
    
	// write output to file: "moments.dat"
	ofstream file2;
	file2.open("moments.dat");
	for (int i = 0; i <= nLevel; i++) {
		file2 << jPlus[i] << "\t" << jMinus[i] << "\t" << hPlus[i] << "\t" << hMinus[i] << "\t" << kPlus[i] << "\t" << kMinus[i] << "\n";
	}
	file2.close();
}

int main()
{
	//Follow particles through simulation
	//Call all functions
	//Read in parameter file

	//creation of host variables
	unsigned int numPhotons;
	cout << "Give an nPhotons value: " << endl;
	cin >> numPhotons;
	const unsigned int nPhotons = numPhotons;
    const int muBins = 10;
	const int nmu = 20;
	int nLevel = 10;
	const int nLev = 21;
	double tauMax = 10;
	int albedo = 1;
	double dTheta = double(1) / double(muBins);
    double theta[muBins] = {0};
	double halfW = 0.5 * dTheta;
    double intensity[nmu] = { 0 };
	double sigmai[nmu] = { 0 };
	double *energy;
	double *erri;
	double *jPlus;
	double *jMinus;
	double *hPlus;
	double *hMinus;
	double *kPlus;
	double *kMinus;
	unsigned int *test;
    
    //initialization of dtheta
    for( int i = 0; i < muBins; i++){
        theta[i] = acos(double(i)*dTheta + halfW)* 180/PI;
    }

	//initialization of host memory
	hPlus = (double*)malloc(nLev * sizeof(double));
	jPlus = (double*)malloc(nLev * sizeof(double));
	kPlus = (double*)malloc(nLev * sizeof(double));
	hMinus = (double*)malloc(nLev * sizeof(double));
	jMinus = (double*)malloc(nLev * sizeof(double));
	kMinus = (double*)malloc(nLev * sizeof(double));
	energy = (double*)malloc(nmu * sizeof(int));
	erri = (double*)malloc(nmu * sizeof(int));
	test = (unsigned int*)malloc(sizeof(unsigned int));

    //Creation of device variables
	double *d_hPlus;
	double *d_kPlus;
	double *d_jMinus;
	double *d_kMinus;
	double *d_hMinus;
	double *d_jPlus;
	double *d_energy;
	double *d_erri;
	unsigned int *d_test;
    
    //Initialization of device memory
	cudaMalloc((void**)&d_hPlus, nLev * sizeof(double));
	cudaMalloc((void**)&d_jPlus, nLev * sizeof(double));
	cudaMalloc((void**)&d_kPlus, nLev * sizeof(double));
	cudaMalloc((void**)&d_hMinus, nLev * sizeof(double));
	cudaMalloc((void**)&d_jMinus, nLev * sizeof(double));
	cudaMalloc((void**)&d_kMinus, nLev * sizeof(double));
	cudaMalloc((void**)&d_energy, nmu * sizeof(double));
	cudaMalloc((void**)&d_erri, nmu * sizeof(double));
	cudaMalloc((void**)&d_test, sizeof(unsigned int));

    //Setting device memory to 0
	cudaMemset(d_hPlus, 0, nLev * sizeof(double));
	cudaMemset(d_jPlus, 0, nLev * sizeof(double));
	cudaMemset(d_kPlus, 0, nLev * sizeof(double));
	cudaMemset(d_hMinus, 0, nLev * sizeof(double));
	cudaMemset(d_jMinus, 0, nLev * sizeof(double));
	cudaMemset(d_kMinus, 0, nLev * sizeof(double));
	cudaMemset(d_energy, 0, nmu * sizeof(double));
	cudaMemset(d_erri, 0, nmu * sizeof(double));
	cudaMemset(d_test, 0, sizeof(unsigned int));

	int seed = static_cast<unsigned int>(time(0));

    for (unsigned int i = 0; i < nPhotons / (SIZE * 512); i++){
        work << <SIZE, 512 >> >(d_hPlus, d_kPlus, d_jMinus, d_kMinus, d_hMinus, d_jPlus,
		muBins, d_energy, nLevel, d_erri, tauMax, albedo, seed, d_test);
        //cudaDeviceSynchronize();
    }
	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(hPlus, d_hPlus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(jPlus, d_jPlus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(kPlus, d_kPlus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hMinus, d_hMinus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(jMinus, d_jMinus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(kMinus, d_kMinus, nLev * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(energy, d_energy, nmu * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(erri, d_erri, nmu * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(test, d_test, sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++) {
		cout << kPlus[i] << endl;
	}
	printf("I ran %u photons", test[0]);

	output(hPlus, kPlus, jMinus, kMinus, hMinus, jPlus, nPhotons, muBins, intensity, energy, nLevel, sigmai, theta, erri);
	cudaFree(d_hPlus);
	cudaFree(d_jPlus);
	cudaFree(d_kPlus);
	cudaFree(d_hMinus);
	cudaFree(d_jMinus);
	cudaFree(d_kMinus);
	cudaFree(d_energy);
	cudaFree(d_erri);
    cudaFree(d_test);
	
	return 0;
}