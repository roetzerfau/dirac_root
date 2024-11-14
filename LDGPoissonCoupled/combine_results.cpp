#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include<algorithm>
#include <math.h>       /* log2 */
namespace fs = std::filesystem;

struct ResultData {
    std::string scenario_name;
    int refinement;
    int p_degree;
    double U, Q, u, q;
};

bool compareResultData(const ResultData& a, const ResultData& b) {
    if (a.p_degree == b.p_degree) {
        return a.refinement < b.refinement;  // Sort by r if p is the same
    }
    return a.p_degree < b.p_degree;  // Otherwise, sort by p
}
void read_csvfile_unsrtd(const fs::path &file_path, std::vector<ResultData> &results) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        ResultData data;
        size_t pos = 0;

        // Parse `scenario_name`
        pos = line.find(';');
        data.scenario_name = line.substr(0, pos);
        line.erase(0, pos + 1);

        // Parse `refinement`
        pos = line.find(';');
        data.refinement = std::stoi(line.substr(0, pos).substr(2)); // skip "r "
        line.erase(0, pos + 1);

        // Parse `p_degree`
        pos = line.find(';');
        data.p_degree = std::stoi(line.substr(0, pos).substr(2)); // skip "p "
        line.erase(0, pos + 1);

        // Parse `U`
        pos = line.find(';');
        data.U = std::stod(line.substr(2, pos)); // skip "U "
        line.erase(0, pos + 1);

        // Parse `Q`
        pos = line.find(';');
        data.Q = std::stod(line.substr(2, pos)); // skip "Q "
        line.erase(0, pos + 1);

        // Parse `u`
        pos = line.find(';');
        data.u = std::stod(line.substr(2, pos)); // skip "u "
        line.erase(0, pos + 1);

        // Parse `q`
        pos = line.find(';');
        data.q = std::stod(line.substr(2, pos)); // skip "q "
        

        results.push_back(data);
    }

    file.close();
     std::sort(results.begin(), results.end(), compareResultData);
}

void write_combined_results(const std::vector<ResultData> &results, const std::string &output_filename) {
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return;
    }

    outfile << "Scenario Name; Refinement; P Degree; U; Q; u; q\n";
 for (size_t i = 0; i < results.size(); i++) {
        const auto &data = results[i];
                // Check if we can calculate the convergence rate
    if (i > 0 && results[i - 1].scenario_name == data.scenario_name && 
        results[i - 1].p_degree == data.p_degree) {
        double rate_U = std::log2(results[i - 1].U / data.U);
        double rate_Q = std::log2(results[i - 1].Q / data.Q);
        double rate_u = std::log2(results[i - 1].u / data.u);
        double rate_q = std::log2(results[i - 1].q / data.q);
  

        outfile << data.scenario_name << "; " 
                << data.refinement << "; " 
                << data.p_degree << "; " 
                << data.U << "(" <<rate_U << ")" <<"; " 
                << data.Q << "(" <<rate_Q << ")" << "; " 
                << data.u << "(" <<rate_u << ")" << "; " 
                << data.q << "(" <<rate_q << ")" << "\n";
                }else
                {
                       outfile << data.scenario_name << "; " 
                << data.refinement << "; " 
                << data.p_degree << "; " 
                << data.U  <<"; " 
                << data.Q  << "; " 
                << data.u  << "; " 
                << data.q  << "\n";
                }
/*


            // Write rates in parentheses next to the values
            outfile << "; " << "Rate U: " << rate_U 
                    << "; " << "Rate Q: " << rate_Q
                    << "; " << "Rate u: " << rate_u
                    << "; " << "Rate q: " << rate_q;*/
      
 }

    outfile.close();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory containing csvfile_unsrtd files>" << std::endl;
        return 1;
    }

    fs::path input_directory = argv[1];
    std::vector<ResultData> combined_results;

    // Traverse directory and read each csvfile_unsrtd
    for (const auto &entry : fs::directory_iterator(input_directory)) {
        if (entry.is_regular_file() && entry.path().string().find("cvg_res_unsrtd") != std::string::npos) {
            read_csvfile_unsrtd(entry.path(), combined_results);
        }
    }

    // Output combined results
    write_combined_results(combined_results, "combined_convergence_results.csv");

    std::cout << "Combined results written to combined_convergence_results.csv" << std::endl;
    return 0;
}