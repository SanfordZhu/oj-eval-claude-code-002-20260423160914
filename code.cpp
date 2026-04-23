#pragma once
#ifndef SJTU_BIGINTEGER
#define SJTU_BIGINTEGER

// Integer 1:
// Implement a signed big integer class that only needs to support simple addition and subtraction

// Integer 2:
// Implement a signed big integer class that supports addition, subtraction, multiplication, and division, and overload related operators

// Do not use any header files other than the following
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Do not use "using namespace std;"

namespace sjtu {
class int2048 {
  // Data members
  bool sign;  // true for positive, false for negative
  std::vector<int> digits;  // digits[0] is least significant

  // Helper functions
  void clean();
  int compare_abs(const int2048 &other) const;
  void add_abs(const int2048 &other);
  void sub_abs(const int2048 &other);
  void mul_fft(const int2048 &other);
  void shift_left(int k);
  void shift_right(int k);
  void div_mod(const int2048 &divisor, int2048 &quotient, int2048 &remainder) const;
  void fft(std::vector<std::complex<double>> &a, bool invert);

public:
  // Constructors
  int2048();
  int2048(long long);
  int2048(const std::string &);
  int2048(const int2048 &);

  // The parameter types of the following functions are for reference only, you can choose to use constant references or not
  // If needed, you can add other required functions yourself
  // ===================================
  // Integer1
  // ===================================

  // Read a big integer
  void read(const std::string &);
  // Output the stored big integer, no need for newline
  void print();

  // Add a big integer
  int2048 &add(const int2048 &);
  // Return the sum of two big integers
  friend int2048 add(int2048, const int2048 &);

  // Subtract a big integer
  int2048 &minus(const int2048 &);
  // Return the difference of two big integers
  friend int2048 minus(int2048, const int2048 &);

  // ===================================
  // Integer2
  // ===================================

  int2048 operator+() const;
  int2048 operator-() const;

  int2048 &operator=(const int2048 &);

  int2048 &operator+=(const int2048 &);
  friend int2048 operator+(int2048, const int2048 &);

  int2048 &operator-=(const int2048 &);
  friend int2048 operator-(int2048, const int2048 &);

  int2048 &operator*=(const int2048 &);
  friend int2048 operator*(int2048, const int2048 &);

  int2048 &operator/=(const int2048 &);
  friend int2048 operator/(int2048, const int2048 &);

  int2048 &operator%=(const int2048 &);
  friend int2048 operator%(int2048, const int2048 &);

  friend std::istream &operator>>(std::istream &, int2048 &);
  friend std::ostream &operator<<(std::ostream &, const int2048 &);

  friend bool operator==(const int2048 &, const int2048 &);
  friend bool operator!=(const int2048 &, const int2048 &);
  friend bool operator<(const int2048 &, const int2048 &);
  friend bool operator>(const int2048 &, const int2048 &);
  friend bool operator<=(const int2048 &, const int2048 &);
  friend bool operator>=(const int2048 &, const int2048 &);
};

// Constants for digit compression
const int BASE = 10000;  // 10^4
const int BASE_DIGITS = 4;

// Helper functions
void int2048::clean() {
  while (!digits.empty() && digits.back() == 0) {
    digits.pop_back();
  }
  if (digits.empty()) {
    sign = true;  // zero is positive
  }
}

int int2048::compare_abs(const int2048 &other) const {
  if (digits.size() != other.digits.size()) {
    return digits.size() < other.digits.size() ? -1 : 1;
  }
  for (int i = (int)digits.size() - 1; i >= 0; --i) {
    if (digits[i] != other.digits[i]) {
      return digits[i] < other.digits[i] ? -1 : 1;
    }
  }
  return 0;
}

void int2048::add_abs(const int2048 &other) {
  int carry = 0;
  size_t max_len = std::max(digits.size(), other.digits.size());
  digits.resize(max_len);
  for (size_t i = 0; i < max_len || carry; ++i) {
    if (i >= digits.size()) digits.push_back(0);
    long long sum = (long long)digits[i] + (i < other.digits.size() ? other.digits[i] : 0) + carry;
    digits[i] = sum % BASE;
    carry = sum / BASE;
  }
}

void int2048::sub_abs(const int2048 &other) {
  int borrow = 0;
  for (size_t i = 0; i < digits.size(); ++i) {
    long long diff = (long long)digits[i] - (i < other.digits.size() ? other.digits[i] : 0) - borrow;
    if (diff < 0) {
      diff += BASE;
      borrow = 1;
    } else {
      borrow = 0;
    }
    digits[i] = diff;
  }
  clean();
}

// FFT for fast multiplication
using cd = std::complex<double>;
const double PI = acos(-1);

void int2048::fft(std::vector<cd> &a, bool invert) {
  int n = a.size();
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = 2 * PI / len * (invert ? -1 : 1);
    cd wlen(cos(ang), sin(ang));
    for (int i = 0; i < n; i += len) {
      cd w(1);
      for (int j = 0; j < len / 2; ++j) {
        cd u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (invert) {
    for (cd &x : a) x /= n;
  }
}

void int2048::mul_fft(const int2048 &other) {
  std::vector<int> a, b;
  a = digits;
  b = other.digits;

  std::vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
  size_t n = 1;
  while (n < a.size() + b.size()) n <<= 1;
  fa.resize(n);
  fb.resize(n);

  fft(fa, false);
  fft(fb, false);
  for (size_t i = 0; i < n; ++i) fa[i] *= fb[i];
  fft(fa, true);

  digits.resize(a.size() + b.size());
  long long carry = 0;
  for (size_t i = 0; i < digits.size(); ++i) {
    long long val = (i < n ? (long long)round(fa[i].real()) : 0) + carry;
    digits[i] = val % BASE;
    carry = val / BASE;
  }
  while (carry) {
    digits.push_back(carry % BASE);
    carry /= BASE;
  }
  clean();
}

// Binary shift for division
void int2048::shift_left(int k) {
  if (k <= 0) return;
  digits.insert(digits.begin(), k, 0);
}

void int2048::shift_right(int k) {
  if (k >= (int)digits.size()) {
    digits.clear();
    return;
  }
  digits.erase(digits.begin(), digits.begin() + k);
}

// Division using binary search
void int2048::div_mod(const int2048 &divisor, int2048 &quotient, int2048 &remainder) const {
  if (divisor.digits.empty()) return;  // Undefined behavior

  quotient = int2048(0);
  remainder = int2048(0);

  for (int i = (int)digits.size() - 1; i >= 0; --i) {
    remainder.shift_left(1);
    remainder.digits[0] = digits[i];
    remainder.clean();

    // Binary search for quotient digit
    int low = 0, high = BASE - 1;
    while (low <= high) {
      int mid = (low + high) / 2;
      int2048 temp = divisor;
      temp.sign = true;
      int2048 product = int2048((long long)mid);
      product *= temp;

      if (product.compare_abs(remainder) <= 0) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    // Append quotient digit at the beginning (most significant)
    quotient.digits.insert(quotient.digits.begin(), high);
    quotient.clean();

    if (high > 0) {
      int2048 temp = divisor;
      temp.sign = true;
      int2048 product = int2048((long long)high);
      product *= temp;
      remainder.sub_abs(product);
    }
  }
  quotient.clean();
  remainder.clean();
}

// Constructors
int2048::int2048() : sign(true), digits(1, 0) {
  clean();
}

int2048::int2048(long long x) {
  if (x < 0) {
    sign = false;
    x = -x;
  } else {
    sign = true;
  }
  if (x == 0) {
    digits.push_back(0);
  } else {
    while (x > 0) {
      digits.push_back(x % BASE);
      x /= BASE;
    }
  }
  clean();
}

int2048::int2048(const std::string &s) {
  size_t start = 0;
  if (s[0] == '-') {
    sign = false;
    start = 1;
  } else if (s[0] == '+') {
    sign = true;
    start = 1;
  } else {
    sign = true;
  }

  // Parse from right to left in chunks of BASE_DIGITS
  for (int i = (int)s.length(); i > (int)start; i -= BASE_DIGITS) {
    int left = std::max((int)start, i - BASE_DIGITS);
    int val = 0;
    for (int j = left; j < i; ++j) {
      val = val * 10 + (s[j] - '0');
    }
    digits.push_back(val);
  }
  clean();
}

int2048::int2048(const int2048 &other) : sign(other.sign), digits(other.digits) {}

// Read a big integer
void int2048::read(const std::string &s) {
  *this = int2048(s);
}

// Output the stored big integer
void int2048::print() {
  if (!sign && !digits.empty()) {
    std::cout << '-';
  }
  if (digits.empty()) {
    std::cout << '0';
    return;
  }
  std::cout << digits.back();
  for (int i = (int)digits.size() - 2; i >= 0; --i) {
    std::cout.fill('0');
    std::cout.width(BASE_DIGITS);
    std::cout << digits[i];
  }
}

// Add a big integer (in-place)
int2048 &int2048::add(const int2048 &other) {
  if (sign == other.sign) {
    add_abs(other);
  } else {
    int cmp = compare_abs(other);
    if (cmp >= 0) {
      sub_abs(other);
    } else {
      int2048 temp = other;
      temp.sub_abs(*this);
      *this = temp;
    }
  }
  clean();
  return *this;
}

// Return the sum of two big integers
int2048 add(int2048 a, const int2048 &b) {
  return a.add(b);
}

// Subtract a big integer (in-place)
int2048 &int2048::minus(const int2048 &other) {
  if (sign == other.sign) {
    int cmp = compare_abs(other);
    if (cmp >= 0) {
      sub_abs(other);
    } else {
      int2048 temp = other;
      temp.sub_abs(*this);
      *this = temp;
      sign = !sign;
    }
  } else {
    add_abs(other);
  }
  clean();
  return *this;
}

// Return the difference of two big integers
int2048 minus(int2048 a, const int2048 &b) {
  return a.minus(b);
}

// Unary operators
int2048 int2048::operator+() const {
  return *this;
}

int2048 int2048::operator-() const {
  int2048 result = *this;
  if (!result.digits.empty()) {
    result.sign = !result.sign;
  }
  return result;
}

// Assignment operator
int2048 &int2048::operator=(const int2048 &other) {
  if (this != &other) {
    sign = other.sign;
    digits = other.digits;
  }
  return *this;
}

// Compound assignment operators
int2048 &int2048::operator+=(const int2048 &other) {
  return add(other);
}

int2048 &int2048::operator-=(const int2048 &other) {
  return minus(other);
}

int2048 &int2048::operator*=(const int2048 &other) {
  if (digits.empty() || other.digits.empty()) {
    *this = int2048(0);
    return *this;
  }
  sign = (sign == other.sign);
  mul_fft(other);
  return *this;
}

int2048 &int2048::operator/=(const int2048 &other) {
  if (other.digits.empty()) return *this;  // Undefined behavior

  int2048 quotient, remainder;
  div_mod(other, quotient, remainder);

  // Floor division: round toward negative infinity
  bool result_sign = (sign == other.sign);
  if (!result_sign && !remainder.digits.empty()) {
    // For negative result with remainder, need to increment absolute quotient
    quotient.add_abs(int2048(1));
  }
  quotient.sign = result_sign;
  *this = quotient;
  return *this;
}

int2048 &int2048::operator%=(const int2048 &other) {
  if (other.digits.empty()) return *this;  // Undefined behavior

  int2048 quotient, remainder;
  div_mod(other, quotient, remainder);

  // Floor division: adjust remainder if needed
  bool result_sign = (sign == other.sign);
  if (!result_sign && !remainder.digits.empty()) {
    // Need to adjust remainder: remainder = remainder - |other|
    int2048 abs_other = other;
    abs_other.sign = true;
    remainder = remainder - abs_other;
  }
  // The remainder's sign should match the divisor's sign
  remainder.sign = other.sign;
  *this = remainder;
  return *this;
}

// Binary operators
int2048 operator+(int2048 a, const int2048 &b) {
  return a += b;
}

int2048 operator-(int2048 a, const int2048 &b) {
  return a -= b;
}

int2048 operator*(int2048 a, const int2048 &b) {
  return a *= b;
}

int2048 operator/(int2048 a, const int2048 &b) {
  return a /= b;
}

int2048 operator%(int2048 a, const int2048 &b) {
  return a %= b;
}

// Stream operators
std::istream &operator>>(std::istream &is, int2048 &x) {
  std::string s;
  is >> s;
  x.read(s);
  return is;
}

std::ostream &operator<<(std::ostream &os, const int2048 &x) {
  if (!x.sign && !x.digits.empty()) {
    os << '-';
  }
  if (x.digits.empty()) {
    os << '0';
    return os;
  }
  os << x.digits.back();
  for (int i = (int)x.digits.size() - 2; i >= 0; --i) {
    os.fill('0');
    os.width(BASE_DIGITS);
    os << x.digits[i];
  }
  return os;
}

// Comparison operators
bool operator==(const int2048 &a, const int2048 &b) {
  if (a.sign != b.sign) return false;
  return a.compare_abs(b) == 0;
}

bool operator!=(const int2048 &a, const int2048 &b) {
  return !(a == b);
}

bool operator<(const int2048 &a, const int2048 &b) {
  if (a.sign != b.sign) return !a.sign;
  if (a.sign) return a.compare_abs(b) < 0;
  return a.compare_abs(b) > 0;
}

bool operator>(const int2048 &a, const int2048 &b) {
  return b < a;
}

bool operator<=(const int2048 &a, const int2048 &b) {
  return !(a > b);
}

bool operator>=(const int2048 &a, const int2048 &b) {
  return !(a < b);
}

} // namespace sjtu

#endif
