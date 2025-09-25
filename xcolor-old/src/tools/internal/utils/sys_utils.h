#ifndef SYS_UTILS_H
#define SYS_UTILS_H

#include <iostream>
#include <time.h>

namespace sys_utils
{

inline bool
float_equal( const float a, const float b )
{
    if ( std::abs( a - b ) < 1e-6 )
        return true;
    else
        return false;
}

inline bool
double_equal( const double a, const double b )
{
    if ( std::abs( a - b ) < 1e-6 )
        return true;
    else
        return false;
}

inline void
PrintWarning( std::string str )
{
    std::cout << "\033[33;40;1m" << str << "\033[0m" << std::endl;
}

inline void
PrintError( std::string str )
{
    std::cout << "\033[31;47;1m" << str << "\033[0m" << std::endl;
}

inline void
PrintInfo( std::string str )
{
    std::cout << "\033[32;40;1m" << str << "\033[0m" << std::endl;
}
}
#endif // SYS_UTILS_H
