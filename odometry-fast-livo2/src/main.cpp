#include "LIVMapper.h"

int main(int argc, char **argv)
{
  LIVMapper mapper; 
  mapper.initializeSubscribersAndPublishers();
  mapper.run();
  return 0;
}