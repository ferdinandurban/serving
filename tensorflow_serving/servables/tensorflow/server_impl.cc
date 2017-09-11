#include "tensorflow_serving/servables/tensorflow/server_impl.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/contrib/session_bundle/bundle_shim.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/servable_handle.h"

namespace tensorflow {
namespace serving {

namespace {

Status ValidateServerRequest(const ServerSpecRequest& request) {
  return tensorflow::Status::OK();
}

}  // namespace

Status ServerManagementImpl::GetServerModels( ServerCore* core, const ServerSpecRequest& request, ServerSpecResponse* response) {
  TF_RETURN_IF_ERROR(ValidateServerRequest(request));
  const std::vector<ServableId> available_servables = core->ListAvailableServableIds();
  std::vector<string> result;

  for (const auto& servable : available_servables) {
    result.push_back(servable.name);
    std::cout << servable.name << std::endl;
  }
  
  
  
  return tensorflow::Status::OK();
}

}  // namespace serving
}  // namespace tensorflow
