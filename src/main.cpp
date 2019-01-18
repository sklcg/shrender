#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "util.hpp"
#include "light.hpp"
#include "mesh.hpp"
#include "render.hpp"
#include "process.hpp"

int main(int argc, char* argv[]) 
{
	Processor processor;
	processor.Process(argc, argv);
	return 0;
}