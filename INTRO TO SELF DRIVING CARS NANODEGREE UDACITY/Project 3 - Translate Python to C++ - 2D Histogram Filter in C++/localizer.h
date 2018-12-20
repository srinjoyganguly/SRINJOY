#ifndef LOCALIZER_H
#define LOCALIZER_H

#include <vector>

// Initializes a grid of beliefs to a uniform distribution. 
std::vector< std::vector <float> > initialize_beliefs(std::vector< std::vector <char> > grid);

/**    
    Implements robot sensing by updating beliefs based on the 
    color of a sensor measurement 
*/
std::vector< std::vector <float> > sense(char color, 
	std::vector< std::vector <char> > grid, 
	std::vector< std::vector <float> > beliefs, 
	float p_hit,
	float p_miss);


/**
    Implements robot motion by updating beliefs based on the 
    intended dx and dy of the robot. 
*/
std::vector< std::vector <float> > move(int dy, int dx, 
	std::vector< std::vector <float> > beliefs,
	float blurring);

#endif /* LOCALIZER_H */