/******************************************************************************
Copyright (c) 2019 Georgia Instititue of Technology
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author : Hyoukjun Kwon (hyoukjun@gatech.edu)
*******************************************************************************/

#include <iostream>
#include <memory>
#include <vector>
#include <list>

#include <boost/program_options.hpp>
#include "BASE_constants.hpp"
#include "BASE_base-objects.hpp"
#include "option.hpp"

#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/scope.hpp>



#include "DFA_tensor.hpp"

#include "AHW-noc-model.hpp"

#include "CA_cost-analysis-results.hpp"

#include "API_configuration.hpp"
#include "API_user-interface-v2.hpp"


#include "DSE_cost-database.hpp"
#include "DSE_design_point.hpp"
#include "DSE_hardware_modules.hpp"
#include "DSE_csv_writer.hpp"



namespace maestro {

  //Singleton objects for common functionalities
	std::shared_ptr<TL::ErrorHandler> error_handler;
	std::shared_ptr<TL::MessagePrinter> message_printer;
	int printout_level = 0;

	void InitializeBaseObjects(int print_lv) {
		error_handler = std::make_shared<TL::ErrorHandler>();
		message_printer = std::make_shared<TL::MessagePrinter>(print_lv);
	}

	void SetPrintOutLevel(int new_lv) {
	  message_printer->SetPrintLv(new_lv);
	}

};


using namespace boost::python;
//long double main_m(int num_pes_t)
list main_m(int num_pes_t)
{
  std::stringstream num_pes_ss; 
  num_pes_ss << "--num_pes=" << num_pes_t;
  int argc_t  = 13; 
  char buffer[100];
  strncpy(buffer,num_pes_ss.str().c_str(),100);
  char* argv_t [] = {
  "./maestro",
  "--print_res=false",
  "--print_res_csv_file=false",
  "--print_log_file=false",
  "--DFSL_file=data/DFSL_description/test.m",
  "--noc_bw=32",
  "--noc_hops=1",
  "--noc_hop_latency=1",
  "--l1_size=13107200",
  "--l2_size=314572800",
  buffer,
  "--print_design_space=false",
  "--msg_print_lv=0"
  } ;
  //for(int i=0; i<13; i++)
  //std::cout << (argv_t[i]) << std::endl;
  maestro::Options option;
  DEBUG_PRINT("In C++ ----> A1 ");
  bool success = option.parse(argc_t, argv_t);
  DEBUG_PRINT("In C++ ----> A2 ");
  if(!success) {
    std::cout << "[MAESTRO] Failed to parse program options" << std::endl;
  }
  
  maestro::InitializeBaseObjects(option.message_print_lv);

  DEBUG_PRINT("In C++ ----> A3 ");

  int num_pes = option.np;
  long double run_time_all = 0;

  list listResult; //python list for passing result to python

  long double runtime_all_layer_sum = 0;
  long double engergy_all_layer_sum = 0;
  long double throughtput_all_layer_sum = 0;
  long double computation_all_layer_sum = 0;
  long double ddr_io_energy_sum = 0;
  /*
   * Hard coded part; will Fix it
   */

  if(option.bw_sweep && option.top_bw_only) {
    int min_bw = option.bw_tick;

    for(int bw = option.min_noc_bw; bw <= option.max_noc_bw; bw += option.bw_tick) {
      std::shared_ptr<std::vector<bool>> noc_multcast = std::make_shared<std::vector<bool>>();
      std::shared_ptr<std::vector<int>> noc_latency = std::make_shared<std::vector<int>>();
      std::shared_ptr<std::vector<int>> noc_bw = std::make_shared<std::vector<int>>();

      if(option.top_bw_only) {
        noc_bw->push_back(bw);
        noc_bw->push_back(70000);
        noc_bw->push_back(70000);
        noc_bw->push_back(70000);
        noc_bw->push_back(70000);
        noc_bw->push_back(70000);

        noc_latency->push_back(option.hop_latency * option.hops);
        noc_latency->push_back(1);
        noc_latency->push_back(1);
        noc_latency->push_back(1);
        noc_latency->push_back(1);
        noc_latency->push_back(1);

        noc_multcast->push_back(option.mc);
        noc_multcast->push_back(true);
        noc_multcast->push_back(true);
        noc_multcast->push_back(true);
        noc_multcast->push_back(true);
        noc_multcast->push_back(true);
      }

      auto config = std::make_shared<maestro::ConfigurationV2>(
          option.dfsl_file_name,
          noc_bw,
          noc_latency,
          noc_multcast,
          option.np,
          option.num_simd_lanes,
          option.bw,
          option.l1_size,
          option.l2_size
          );

      std::cout << "BW: " << bw << std::endl;
      auto api = std::make_shared<maestro::APIV2>(config);
      auto res = api->AnalyzeNeuralNetwork(option.print_res_to_screen, true);

    }
  }
  else {
    DEBUG_PRINT("In C++ ----> A4 ");
    std::shared_ptr<std::vector<bool>> noc_multcast = std::make_shared<std::vector<bool>>();
    std::shared_ptr<std::vector<int>> noc_latency = std::make_shared<std::vector<int>>();
    std::shared_ptr<std::vector<int>> noc_bw = std::make_shared<std::vector<int>>();

    noc_bw->push_back(option.bw);
    noc_bw->push_back(option.bw);
    noc_bw->push_back(option.bw);
    noc_bw->push_back(option.bw);


    noc_latency->push_back(option.hop_latency * option.hops);
    noc_latency->push_back(option.hop_latency * option.hops);
    noc_latency->push_back(option.hop_latency * option.hops);
    noc_latency->push_back(option.hop_latency * option.hops);

    noc_multcast->push_back(true);
    noc_multcast->push_back(true);
    noc_multcast->push_back(true);
    noc_multcast->push_back(true);

    DEBUG_PRINT("In C++ ----> A5 ");
    auto config = std::make_shared<maestro::ConfigurationV2>(
        option.dfsl_file_name,
        noc_bw,
        noc_latency,
        noc_multcast,
        option.np,
        option.num_simd_lanes,
        option.bw,
        option.l1_size,
        option.l2_size
        );

    DEBUG_PRINT("In C++ ----> A6 ");
    auto api = std::make_shared<maestro::APIV2>(config);
    DEBUG_PRINT("In C++ ----> A7--> ");
    auto res = api->AnalyzeNeuralNetwork(option.print_res_to_screen, option.print_res_to_csv_file, option.print_log_file);
    DEBUG_PRINT("In C++ ----> A8 ");
    for(auto& layer_res : *res) {
      auto upper_most_cluster_res = layer_res->at(layer_res->size()-1);
      //std::cout<<"zbr:"<<upper_most_cluster_res->GetRuntime();
      run_time_all += upper_most_cluster_res->GetRuntime();
    }

    int ly_i = 0;
    for(auto& rt : *(api->runtime_all_layer)) {
      runtime_all_layer_sum += rt;
      engergy_all_layer_sum += api->engergy_all_layer->at(ly_i);
      throughtput_all_layer_sum += api->throughtput_all_layer->at(ly_i);
      computation_all_layer_sum += api->computation_all_layer->at(ly_i);
      ddr_io_energy_sum += api->ddr_io_energy->at(ly_i);

      ly_i ++;
    }

    //print("[ 0:runtime, 1:engergy, 2:throughtput, 3:computation, 4:l1_size, 5:l2_size, 6:area, 7:power, 8:ddr_energy, 9:num_pe_utilized, 10:reuse_input, 11:reuse_weight, 12:reuse_output]")
    engergy_all_layer_sum = ddr_io_energy_sum + engergy_all_layer_sum; 
    //engergy_all_layer_sum = engergy_all_layer_sum; 
    listResult.append(runtime_all_layer_sum);
    listResult.append(engergy_all_layer_sum);
    listResult.append(throughtput_all_layer_sum);
    listResult.append(computation_all_layer_sum);
    listResult.append(api->l1_size);
    listResult.append(api->l2_size);
    listResult.append(api->area);
    listResult.append(api->power);
    listResult.append(ddr_io_energy_sum);
    listResult.append(api->num_pe_utilized);
    listResult.append(api->reuse_factor_all->at(0));
    listResult.append(api->reuse_factor_all->at(1));
    listResult.append(api->reuse_factor_all->at(2));
    listResult.append(api->l2_renergy* maestro::DSE::cost::mac_energy);
    listResult.append(api->l2_wenergy* maestro::DSE::cost::mac_energy);
    listResult.append(api->l1_renergy* maestro::DSE::cost::mac_energy);
    listResult.append(api->l1_wenergy* maestro::DSE::cost::mac_energy);
    listResult.append(api->mac_energy);
  }
  return listResult;
//  return 0;
}



BOOST_PYTHON_MODULE(maestro)
{
    register_exception_translator<maestro::TL::my_exception>(&maestro::TL::translate);
    def("main_m", main_m);
    class_<maestro::Options>("Options");
    enum_<maestro::DataClass>("DataClass");
    enum_<maestro::TL::ErrorCode>("ErrorCode");
    class_<maestro::TL::ErrorHandler>("ErrorHandler");
    class_<maestro::TL::MessagePrinter>("MessagePrinter", init<int>());
    def("InitializeBaseObjects", &maestro::InitializeBaseObjects);
    //class_<maestro::ConfigurationV2>("ConfigurationV2", init<>);

}

