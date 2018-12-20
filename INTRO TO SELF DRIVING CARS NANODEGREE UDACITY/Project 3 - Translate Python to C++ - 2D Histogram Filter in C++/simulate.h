#ifndef SIMULATE_H
#define SIMULATE_H

#include <vector>

class Simulation {
	
private:
	std::vector <char> get_colors();

public: 
	std::vector < std::vector <char> > grid;
	std::vector < std::vector <float> > beliefs;

	float blur, p_hit, p_miss, incorrect_sense_prob;

	int height, width, num_colors;
	
	std::vector<int> true_pose;
	std::vector<int> prev_pose;

	std::vector <char> colors;
	Simulation(std::vector < std::vector<char> >, float, float, std::vector <int>);

};

#endif /* SIMULATE_H */