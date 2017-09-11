#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow_serving/apis/server.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace tensorflow {
namespace serving {

// Utility methods for implementation of PredictionService::Predict.
class ServerManagementImpl {
 public:
  
  Status GetServerModels(const RunOptions& run_options, ServerCore* core,
                 const ServerSpecRequest& request, ServerSpecREsponse* response);

 private:
  
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_PREDICT_IMPL_H_
