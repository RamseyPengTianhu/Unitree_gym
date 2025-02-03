
all:
	@(cd sipgen; $(MAKE))
	@(cd siplib; $(MAKE))

install:
	@(cd sipgen; $(MAKE) install)
	@(cd siplib; $(MAKE) install)
	@test -d $(DESTDIR)/home/tianhu/anaconda3/envs/humanoid_exp/lib/python3.8/site-packages || mkdir -p $(DESTDIR)/home/tianhu/anaconda3/envs/humanoid_exp/lib/python3.8/site-packages
	cp -f sipconfig.py $(DESTDIR)/home/tianhu/anaconda3/envs/humanoid_exp/lib/python3.8/site-packages/sipconfig.py
	cp -f /home/tianhu/expressive-humanoid/ASE/ase/poselib/sip-4.19.3/sipdistutils.py $(DESTDIR)/home/tianhu/anaconda3/envs/humanoid_exp/lib/python3.8/site-packages/sipdistutils.py

clean:
	@(cd sipgen; $(MAKE) clean)
	@(cd siplib; $(MAKE) clean)
