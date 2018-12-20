#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <string>

// Normalizes a grid of numbers. 
std::vector< std::vector<float> > normalize(std::vector< std::vector <float> > grid);

/** 
	Blurs (and normalizes) a grid of probabilities by spreading 
	probability from each cell over a 3x3 "window" of cells. This 
	function assumes a cyclic world where probability "spills 
	over" from the right edge to the left and bottom to top.
*/
std::vector < std::vector <float> > blur(std::vector < std::vector < float> > grid, float blurring);

/**
    Determines when two grids of floating point numbers 
    are "close enough" that they should be considered 
    equal. Useful for battling "floating point errors".
*/
bool close_enough(std::vector < std::vector <float> > g1, std::vector < std::vector <float> > g2);

bool close_enough(float v1, float v2);

// Helper function for reading in map data
std::vector <char> read_line(std::string s);

// Helper function for reading in map data
std::vector < std::vector <char> > read_map(std::string file_name);

// Creates a grid of zeros
std::vector < std::vector <float> > zeros(int height, int width);

#endif /* HELPERS_H */