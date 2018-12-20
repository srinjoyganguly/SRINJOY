#ifndef DEBUGGING_HELPERS_H
#define DEBUGGING_HELPERS_H

#include <vector>

// Displays a grid of beliefs. Does not return.
void show_grid(std::vector < std::vector <float> > grid);

// Displays a grid map of the world
void show_grid(std::vector < std::vector <char> > map);

#endif /* DEBUGGING_HELPERS_H */