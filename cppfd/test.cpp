#include<iostream>
#include<array>
#include"mesh.cpp"

int main(int argc, char const *argv[]) {
  Mesh m;
  m.set_origin(0.1, 0.5);
  auto o = m.get_origin();
  std::cout<<o[0]<<" "<<o[1]<<std::endl;
  auto c = m.index_to_cartesian(6, 5, 25);
  std::cout<<c[0]<<" "<<c[1]<<std::endl;
  auto i = m.cartesian_to_index(c[0], c[1], 5, 5);
  std::cout<<i<<std::endl;
  return 0;
}
