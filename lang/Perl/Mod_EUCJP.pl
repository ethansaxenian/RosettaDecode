package Mod_EUCJP;
no warnings "deprecated";
use encoding "euc-jp";
sub new {
  my $class = shift;
  my $str = shift || qw;
  my $self = bless { 
      str => '',
  }, $class;
  $self->set($str);
  $self;
}
sub set {
  my ($self,$str) = @_;
  $self->{str} = $str;
  $self;
}
sub str { shift->{str}; }
sub put { print shift->{str}; }
1;
__END__
