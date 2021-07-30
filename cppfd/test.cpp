#include<iostream>
#include<array>
#include"mesh.cpp"

int main(int argc, char const *argv[]) {
  Mesh m;
  m.set_origin(0.1, 0.5);
  auto o = m.get_origin();
  std::cout<<o[0]<<" "<<o[1]<<std::endl;
  return 0;
}
