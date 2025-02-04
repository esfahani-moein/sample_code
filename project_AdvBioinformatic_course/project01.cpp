#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

const int MATCH = 1;      // Score for matching characters
const int MISMATCH = -1;  // Penalty for mismatching characters
const int GAP = -2;       // Penalty for gaps

std::pair<std::string, std::string> needlemanWunsch(const std::string& seq1, const std::string& seq2) {
    int m = seq1.length();
    int n = seq2.length();
    
    std::vector<std::vector<int>> score(m + 1, std::vector<int>(n + 1));
    std::vector<std::vector<char>> traceback(m + 1, std::vector<char>(n + 1));
    
    
    for(int i = 0; i <= m; i++) {
        score[i][0] = i * GAP;
        traceback[i][0] = 'u'; 
    }
    for(int j = 0; j <= n; j++) {
        score[0][j] = j * GAP;
        traceback[0][j] = 'l'; 
    }
    
  
    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            int match = score[i-1][j-1] + (seq1[i-1] == seq2[j-1] ? MATCH : MISMATCH);
            int del = score[i-1][j] + GAP;
            int ins = score[i][j-1] + GAP;
            
            score[i][j] = std::max({match, del, ins});
            
            if(score[i][j] == match) traceback[i][j] = 'd'; 
            else if(score[i][j] == del) traceback[i][j] = 'u'; 
            else traceback[i][j] = 'l'; 
        }
    }
    
    std::string aligned1, aligned2;
    int i = m, j = n;
    
    while(i > 0 || j > 0) {
        if(i > 0 && j > 0 && traceback[i][j] == 'd') {
            aligned1 = seq1[i-1] + aligned1;
            aligned2 = seq2[j-1] + aligned2;
            i--; j--;
        }
        else if(i > 0 && traceback[i][j] == 'u') {
            aligned1 = seq1[i-1] + aligned1;
            aligned2 = '-' + aligned2;
            i--;
        }
        else {
            aligned1 = '-' + aligned1;
            aligned2 = seq2[j-1] + aligned2;
            j--;
        }
    }
    
    return {aligned1, aligned2};
}

int main() {
    std::string seq1 = "GCATGCU";
    std::string seq2 = "GATTACA";
    
    auto [aligned1, aligned2] = needlemanWunsch(seq1, seq2);
    
    std::cout << "Sequence 1: " << aligned1 << std::endl;
    std::cout << "Sequence 2: " << aligned2 << std::endl;
    
    return 0;
}