#include <unistd.h>
#include "example.skel.h"

int main()
{
      struct example *skel;
      int err = 0;

      skel = example__open();
      if (!skel)
              goto cleanup;

      err = example__load(skel);
      if (err)
              goto cleanup;

      err = example__attach(skel);
      if (err)
              goto cleanup;

      pause();

cleanup:
      example__destroy(skel);
      return err;
}
