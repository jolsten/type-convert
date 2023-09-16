static signed long long twoscomp(unsigned long long uint,  unsigned char size) {
    unsigned char max_pos_val = 1;
    unsigned char pad_bits;
    union {
        unsigned long long int u;
        signed long long int s;
    } tmp;

    tmp.u = uint;

    // Determine max positive value for 2c int of this size
    max_pos_val = ((max_pos_val << (size-1))) - 1;

    // If there is a leading 0b1 (i.e. > max pos val),
    // then the number is negative, and the 2c value must
    // be obtained
    if (tmp.u > max_pos_val) {
        // Determine size of long long on this system,
        // to determine needed pad bits
        pad_bits = sizeof(tmp.u) * 8 - size;

        tmp.u = tmp.u << pad_bits;
        tmp.s = tmp.s >> pad_bits;
    }

    return tmp.s;
}